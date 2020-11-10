#include "stretch_shear_forces.h"

#include <igl/sparse_cached.h>

using namespace std;
using namespace Eigen;


namespace  {

typedef Triplet<double> Tri;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

bool isPSD(const MatrixXd& A) {
    assert( (A - A.transpose()).norm() < 1e-8 );
    return A.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff() >= -1e-10;
}
}

StretchShear::StretchShear() {}

void StretchShear::init(double k_stretch, double k_stretch_damping, double k_shear, double k_shear_damping, MatrixXd & V, MatrixXi & F) {
    this->k_stretch = k_stretch;
    this->k_stretch_damping = k_stretch_damping;
    this->k_shear = k_shear;
    this->k_shear_damping = k_shear_damping;
    this->F = F;

    n = V.rows();
    m = F.rows();
    n3 = 3 * n;
    m81 = 81 * m;  // per face: 9 matrices * 9 entries matrix entries

    precompute_rest_shape(V);
}

void StretchShear::precompute_rest_shape(MatrixXd& V) {
    // triangle area and V_inverse of the reference configuration
    a.resize(m);
    V_inverse = vector<Matrix2d>(m);
    B = vector<MatrixXd>(m, MatrixXd(3,2));
    
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
        Vector3d v1 = V.row(F(i, 1)) - V.row(F(i, 0));
        Vector3d v2 = V.row(F(i, 2)) - V.row(F(i, 0));
        a(i) = 0.5f * v1.cross(v2).norm();
        
        // use a u,v coordinate frame instead
        Vector3d u = v1.normalized();
        Vector3d normal = u.cross(v2);
        Vector3d w = normal.cross(u).normalized();
        Matrix2d Vref; Vref << v1.norm(), v2.dot(u), 0, v2.dot(w);
        V_inverse[i] = Vref.inverse();
        
        // some mysterious needed precomp (for dJ/dv_x), left col for J_x, right col for J_y
        B[i](0, 0) = -V_inverse[i](0, 0) - V_inverse[i](1, 0);
        B[i](0, 1) = -V_inverse[i](0, 1) - V_inverse[i](1, 1);
        B[i].block(1, 0, 2, 2) = V_inverse[i];
    }
}

const Eigen::VectorXd& StretchShear::getStretch() const
{
    return this->S;
}

const Eigen::SparseMatrix<double>& StretchShear::getK() const
{
    return K;
}

const Eigen::SparseMatrix<double>& StretchShear::getD() const
{
    return D;
}

void StretchShear::compute_forces(
    const MatrixXd & X,        // in: vertex positions
    const VectorXd & V,        // in: velocitiy at vertex positions
    VectorXd & Force)        // out: force vector
{
    // initialize triplet data
    if(!tripletsInitialized)
    {
        std::vector<Eigen::Triplet<double>> tri;
        
        for (int f = 0; f < m; f++)
        {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j <= i; j++) {    // go through all pairs of vertices
                    for (int p = 0; p < 3; p++) {        // x, y, z
                        for (int q = 0; q <= p; q++) {    // x, y, z
                            int idi = F(f, i);            // we always want F(f,i) < F(f,j) here, such that we only fill the lower triangular matrix
                            int idj = F(f, j);
                            int idp = p;                // we might need to swap p and q, since we might need to use K_ji.transpose()
                            int idq = q;                // K_ij is not symmetric here
                            if (F(f, i) > F(f, j)) {
                                idi = F(f, j);
                                idj = F(f, i);
                                idp = q;
                                idq = p;
                            }
                            
                            int row = idi + p * n; // corresponds to vertex id (i/j) and shifted by number of vertices for y and z coordinates (p)
                            int col = idj + q * n;
                            
                            tri.push_back(Tri(row, col, 0));            // ... these are always in the lower triangle
                           
                            if (i != j) {
                                row = idj + p * n;                                                         // only i and j are switched
                                col = idi + q * n;
                                tri.push_back(Tri(row, col, 0));        // K_ij = K_ji.transpose() and symmetric!
                            }
                        }
                    }
                }
            }
        }
        
        K = SparseMatrix<double>(n3, n3);
        igl::sparse_cached_precompute(tri, K_data, K);
        D = K;

        triK.resize(tri.size());
        triD.resize(tri.size());
        triF.resize(9 * m);
        S.resize(m);
        
        tripletsInitialized = true;
    }
    
    triF.setZero();

    #pragma omp parallel for
    for (int f = 0; f < m; f++) {
        MatrixXd D(3, 2);
        D.col(0) = X.row(F(f, 1)) - X.row(F(f, 0));
        D.col(1) = X.row(F(f, 2)) - X.row(F(f, 0));
        
        MatrixXd J = D * V_inverse[f];            // deformation gradient
        Vector2d Jnorm = J.colwise().norm();
        MatrixXd Jnormalized = J;
        Jnormalized.col(0) /= Jnorm(0);
        Jnormalized.col(1) /= Jnorm(1);
        
        vector< Vector3d > v(3);                // velocity
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                v[i](j) = V(F(f, i) + j * n);
        
        Matrix9d K_stretch9;
        Matrix9d D_stretch9;
        
        Matrix9d K_shear9;
        Matrix9d D_shear9;
    
        // Begin stretch
        {
            // adjust for different mesh resolutions                -> TODO precompute!
            double k_stretch_f = k_stretch / a(f);
            
            // condition vector
            // from [Large Steps in Cloth Simulation, Baraff & Witkin, 1998] - Equation 10
            Vector2d C_stretch = a(f) * (Jnorm - Vector2d::Ones());
            S(f) = Jnorm.maxCoeff();
            
            // --- first order derivatives
            vector<MatrixXd> dC_dv(3);            // derivative of C_stretch by all 3 vertices
            Vector2d Cdot = Vector2d::Zero();    // derivative times velocity
            for (int i = 0; i < 3; i++) {        // go through each vertex
                dC_dv[i].resize(3, 2);
                dC_dv[i].col(0) = a(f) * B[f](i, 0) * Jnormalized.col(0);        // this derivative was checked numerically - correct for dv0x!
                dC_dv[i].col(1) = a(f) * B[f](i, 1) * Jnormalized.col(1);
                // See: Implementing Baraff & Witkin's Cloth Simulation by Pritchard:
                Cdot += dC_dv[i].transpose() * v[i];
            }
            
            for (int i = 0; i < 3; i++) {
                // compute forces
                Vector3d F_stretch = -k_stretch_f * (dC_dv[i] * C_stretch);
                
                // compute damping
                Vector3d D_stretch = -k_stretch_damping * dC_dv[i] *  Cdot;
                F_stretch += D_stretch;
                
                // force + damping - put into the right position of the global force vector
                for (int j = 0; j < 3; j++)
                    triF(9 * f + 3 * i + j) += F_stretch(j);
            }
            
            // --- second order derivatives
            // compute stiffness matrix elements
            Matrix3d IJx = a(f) * (Matrix3d::Identity() - Jnormalized.col(0) * Jnormalized.col(0).transpose() ) / Jnorm(0);    // symmetric
            Matrix3d IJy = a(f) * (Matrix3d::Identity() - Jnormalized.col(1) * Jnormalized.col(1).transpose() ) / Jnorm(1);    // symmetric
            
            
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j <= i; j++) {    // go through all pairs of vertices
                    Matrix3d dCx_dvdv = B[f](i, 0) * B[f](j, 0) * IJx;        // this Hessian was numerically checked for dv0x_dv0x - correct
                    Matrix3d dCy_dvdv = B[f](i, 1) * B[f](j, 1) * IJy;        // is is also the same as in Implementing Baraff & Witkin's Cloth Simulation by Pritchard
                    
                    // stiffness matrix
                    Matrix3d K_stretch = - k_stretch_f * (dC_dv[i] * dC_dv[j].transpose() + dCx_dvdv * C_stretch(0) + dCy_dvdv * C_stretch(1));    // symmetric
                    
                    // compute damping
                    Matrix3d KD = dCx_dvdv * Cdot(0) + dCy_dvdv * Cdot(1);
                    K_stretch += -k_stretch_damping * KD;
                    
                    // damping elements
                    Matrix3d D_stretch = -k_stretch_damping * dC_dv[i] * dC_dv[j].transpose();
                    
                    K_stretch9.block(3 * i, 3 * j, 3, 3) = K_stretch;        // we only fill the lower triangular matrix here
                    D_stretch9.block(3 * i, 3 * j, 3, 3) = D_stretch;
                }
            }
        }
        
        // End stretch
        // Begin shear
        {
            vector< Vector3d > v(3);                // velocity
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    v[i](j) = V(F(f, i) + j * n);
            
            // adjust for different mesh resolutions
            double k_shear_f = k_shear / a(f);
            
            // condition vector
            // from [Large Steps in Cloth Simulation, Baraff & Witkin, 1998] - Section 4.3
            double C_shear = a(f) * J.col(0).transpose() * J.col(1);
            
            // first order derivatives
            vector<Vector3d> C_dv(3);
            double Cdot = 0;            // derivative times velocity
            for (int i = 0; i < 3; i++) {
                C_dv[i] = a(f) * (B[f](i, 0) * J.col(1) + B[f](i, 1) * J.col(0));
                Cdot += C_dv[i].transpose() * v[i];
            }
            for (int i = 0; i < 3; i++) {
                // compute forces
                Vector3d F_shear = -k_shear_f * (C_dv[i] * C_shear);
                
                // compute damping
                Vector3d D_shear = -k_shear_damping * C_dv[i] * Cdot;
                
                // force + damping - put into the right position of the global force vector
                for (int j = 0; j < 3; j++)
                    triF(9 * f + 3 * i + j) += F_shear(j) + D_shear(j);
                  //   Force(F(f, i) + j * n) += F_shear(j) + D_shear(j);
            }
            
            // second order derivatives

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j <= i; j++) {    // go through all pairs of vertices
                    // compute stiffness elements
                    // this is a diagonal matrix with the same values along the diagonal.
                    // So we can just multiply with the double instead afterwards to save comp. time
                    double dC_dvdv = a(f) * (B[f](i, 0) * B[f](j, 1) + B[f](i, 1) * B[f](j, 0));
                    
                    Matrix3d K_shear = C_dv[i] * C_dv[j].transpose();
                    K_shear.diagonal() += (dC_dvdv * C_shear) * Vector3d::Ones();
                    K_shear *= -k_shear_f;
                    
                    // compute damping
                    Matrix3d KD = -k_shear_damping * dC_dvdv * Cdot * Matrix3d::Identity();
                    K_shear += KD;
                    
                    // damping elements
                    Matrix3d D_shear = -k_shear_damping * C_dv[i] * C_dv[j].transpose();
                    K_shear9.block(3 * i, 3 * j, 3, 3) = K_shear;        // we only fill the lower triangular matrix here
                    D_shear9.block(3 * i, 3 * j, 3, 3) = D_shear;
                }
            }
        }
        
        // End shear
                
        int tripletOffset = f * 54;
        
        Matrix9d K9 = K_stretch9 + K_shear9;
        Matrix9d D9 = D_stretch9 + D_shear9;
       
        SelfAdjointEigenSolver<MatrixXd> eig(-K9);
        VectorXd eigenvalues = eig.eigenvalues();
        if (eigenvalues.minCoeff() < -1e-10) {
            MatrixXd eigenvectors = eig.eigenvectors();
            MatrixXd S = MatrixXd::Zero(eigenvalues.rows(), eigenvalues.rows());
            for (int s = 0; s < eigenvalues.rows(); s++)
                S(s, s) = eigenvalues(s) > -1e-10 ? eigenvalues(s) : 0;
            K9 = -eigenvectors * S * eigenvectors.transpose();
        }
        
        // create triplets - only for the lower triangle of the whole matrices K and D
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {    // go through all pairs of vertices
                for (int p = 0; p < 3; p++) {        // x, y, z
                    for (int q = 0; q <= p; q++) {    // x, y, z
                        int idi = F(f, i);            // we always want F(f,i) < F(f,j) here, such that we only fill the lower triangular matrix
                        int idj = F(f, j);
                        int idp = p;                // we might need to swap p and q, since we might need to use K_ji.transpose()
                        int idq = q;                // K_ij is not symmetric here
                        if (F(f, i) > F(f, j)) {
                            idi = F(f, j);
                            idj = F(f, i);
                            idp = q;
                            idq = p;
                        }
                        
                        triK[tripletOffset] = K9(3*i + idp, 3*j + idq);
                        triD[tripletOffset] = D9(3*i + idp, 3*j + idq);
                        ++tripletOffset;
                        
                        
                        if (i != j) {
                            triK[tripletOffset] = K9(3*i + idq, 3*j + idp);
                            triD[tripletOffset] = D9(3*i + idq, 3*j + idp);
                            ++tripletOffset;
                        }
                    }
                }
            }
        }
    }
    
    
    // fill force
    for(int f = 0; f < m; ++f)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Force(F(f, i) + j * n) += triF(9 * f + 3 * i + j);
    }
    
    // build the sparse matrices from triplets
    igl::sparse_cached(triK, K_data, this->K);
    igl::sparse_cached(triD, K_data, this->D);
}
