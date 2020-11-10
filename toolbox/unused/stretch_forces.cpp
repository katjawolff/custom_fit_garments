#include "stretch_forces.h"

#include <igl/sparse_cached.h>

using namespace std;
using namespace Eigen;

typedef Triplet<double> Tri;

bool isPSD(const MatrixXd& A) {
    assert( (A - A.transpose()).norm() < 1e-8 );
    return A.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff() >= -1e-10;
}

Stretch::Stretch() {}

void Stretch::init(double k_stretch, double k_damping, MatrixXd & V, MatrixXi & F) {
	this->k_stretch = k_stretch;
	this->k_damping = k_damping;
	this->F = F;

	n = V.rows();
	m = F.rows();
	n3 = 3 * n;
	m81 = 81 * m;  // per face: 9 matrices * 9 entries matrix entries

    triK.reserve(m81);
    triD.reserve(m81);
    
	precompute_rest_shape(V);
}

void Stretch::precompute_rest_shape(MatrixXd& V) {
	// triangle area and V_inverse of the reference configuration
	a.resize(m);
	V_inverse = vector<Matrix2d>(m);
	B = vector<MatrixXd>(m);
	for (int i = 0; i < m; i++) {
		Vector3d v1 = V.row(F(i, 1)) - V.row(F(i, 0));
		Vector3d v2 = V.row(F(i, 2)) - V.row(F(i, 0));
		a(i) = 0.5f * v1.cross(v2).norm();

		// use a u,v coordinate frame instead
		Vector3d u = v1.normalized();
		Vector3d normal = u.cross(v2);
		Vector3d w = normal.cross(u).normalized();
		//Matrix2d Vref; Vref << v1.norm(), v2.dot(u), 0, v2.dot(w);
		//V_inverse[i] = Vref.inverse();
		double v1_norm = 1. / v1.norm();
		double v2_dot_u = v2.dot(u);
		double v2_dot_w = v2.dot(w);
		V_inverse[i] << v1_norm, -v1_norm * v2_dot_u / v2_dot_w, 0, 1. / v2_dot_w;

		// some mysterious needed precomp (for dJ/dv_x), left col for J_x, right col for J_y
		B[i].resize(3, 2);
		B[i](0, 0) = -V_inverse[i](0, 0) - V_inverse[i](1, 0);
		B[i](0, 1) = -V_inverse[i](0, 1) - V_inverse[i](1, 1);
		B[i].block(1, 0, 2, 2) = V_inverse[i];
	}
}

void Stretch::compute_stretch(VectorXd & S){
	S = this->S;
}

const Eigen::SparseMatrix<double>& Stretch::getK() const
{
    return K;
}

const Eigen::SparseMatrix<double>& Stretch::getD() const
{
    return D;
}

void Stretch::compute_forces(
	const MatrixXd & X,		// in: vertex positions
	const VectorXd & V,		// in: velocitiy at vertex positions
	VectorXd & Force)		// out: force vector
{
	Force = VectorXd::Zero(3 * n);
	S = VectorXd(m);

	// only do calculations if k != 0
	if (k_stretch == 0) {
		K = SparseMatrix<double>(n3, n3);	// zero matrices
		D = SparseMatrix<double>(n3, n3);
		return;
	}
    
    // initialize triplet data
    if(!tripletsInitialized)
    {
        triK.clear();
        triD.clear();
        
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

                            triK.push_back(Tri(row, col, 0));            // ... these are always in the lower triangle
                            triD.push_back(Tri(row, col, 0));
                            if (i != j) {
                                row = idj + p * n;                                                         // only i and j are switched
                                col = idi + q * n;
                                triK.push_back(Tri(row, col, 0));        // K_ij = K_ji.transpose() and symmetric!
                                triD.push_back(Tri(row, col, 0));        // but D_ij is not symmetric, why we need to do it this whole complicated way
                            }
                        }
                    }
                }
            }
        }
        
        this->K = SparseMatrix<double>(n3, n3);
        this->D = SparseMatrix<double>(n3, n3);
        igl::sparse_cached_precompute(triK, K_data, this->K);
        igl::sparse_cached_precompute(triD, D_data, this->D);
        
        tripletsInitialized = true;
    }
    
    
//#pragma omp parallel
	for (int f = 0; f < m; f++) {
		MatrixXd D(3, 2);
		D.col(0) = X.row(F(f, 1)) - X.row(F(f, 0));
		D.col(1) = X.row(F(f, 2)) - X.row(F(f, 0));

		MatrixXd J = D * V_inverse[f];			// deformation gradient
		Vector2d Jnorm = J.colwise().norm();
		MatrixXd Jnormalized = J;
		Jnormalized.col(0) /= Jnorm(0);
		Jnormalized.col(1) /= Jnorm(1);

		vector< Vector3d > v(3);				// velocity
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				v[i](j) = V(F(f, i) + j * n);

		// adjust for different mesh resolutions				-> TODO precompute!
		double k_stretch_f = k_stretch / a(f);

		// condition vector
		// from [Large Steps in Cloth Simulation, Baraff & Witkin, 1998] - Equation 10
		Vector2d C_stretch = a(f) * (Jnorm - Vector2d::Ones());
		S(f) = Jnorm.maxCoeff();

		// --- first order derivatives		
		vector<MatrixXd> dC_dv(3);			// derivative of C_stretch by all 3 vertices
		Vector2d Cdot = Vector2d::Zero();	// derivative times velocity
		for (int i = 0; i < 3; i++) {		// go through each vertex
			dC_dv[i].resize(3, 2);
			dC_dv[i].col(0) = a(f) * B[f](i, 0) * Jnormalized.col(0);		// this derivative was checked numerically - correct for dv0x!
			dC_dv[i].col(1) = a(f) * B[f](i, 1) * Jnormalized.col(1);
			// See: Implementing Baraff & Witkin's Cloth Simulation by Pritchard:
			Cdot += dC_dv[i].transpose() * v[i];
		}

		for (int i = 0; i < 3; i++) {
			// compute forces
			Vector3d F_stretch = -k_stretch_f * (dC_dv[i] * C_stretch);

			// compute damping
			Vector3d D_stretch = -k_damping * dC_dv[i] *  Cdot;
			F_stretch += D_stretch;

			// force + damping - put into the right position of the global force vector
			for (int j = 0; j < 3; j++) 
				Force(F(f, i) + j * n) += F_stretch(j);
		}			

		// --- second order derivatives
		// compute stiffness matrix elements
		Matrix3d IJx = a(f) * (Matrix3d::Identity() - Jnormalized.col(0) * Jnormalized.col(0).transpose() ) / Jnorm(0);	// symmetric
		Matrix3d IJy = a(f) * (Matrix3d::Identity() - Jnormalized.col(1) * Jnormalized.col(1).transpose() ) / Jnorm(1);	// symmetric

		MatrixXd K_stretch9(9, 9);
		MatrixXd D_stretch9(9, 9);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j <= i; j++) {	// go through all pairs of vertices
				Matrix3d dCx_dvdv = B[f](i, 0) * B[f](j, 0) * IJx;		// this Hessian was numerically checked for dv0x_dv0x - correct
				Matrix3d dCy_dvdv = B[f](i, 1) * B[f](j, 1) * IJy;		// is is also the same as in Implementing Baraff & Witkin's Cloth Simulation by Pritchard

				// stiffness matrix
				Matrix3d K_stretch = - k_stretch_f * (dC_dv[i] * dC_dv[j].transpose() + dCx_dvdv * C_stretch(0) + dCy_dvdv * C_stretch(1));	// symmetric

				// compute damping 
				Matrix3d KD = dCx_dvdv * Cdot(0) + dCy_dvdv * Cdot(1);
				K_stretch += -k_damping * KD;
				
				// damping elements
				Matrix3d D_stretch = -k_damping * dC_dv[i] * dC_dv[j].transpose();

				K_stretch9.block(3 * i, 3 * j, 3, 3) = K_stretch;		// we only fill the lower triangular matrix here
				D_stretch9.block(3 * i, 3 * j, 3, 3) = D_stretch;			
			}			
		}

		// project onto positive eigenvalues
		// these 9x9 matrices are real symmetric => diagonizable (spectral theorem)
		// symmetry also gives orthogonal eigenvalues, therefore Eval.transpose() = Eval.inverse()
		SelfAdjointEigenSolver<MatrixXd> eig(-K_stretch9);
		VectorXd eigenvalues = eig.eigenvalues();
		if (eigenvalues.minCoeff() < -1e-10) {
			MatrixXd eigenvectors = eig.eigenvectors();
			MatrixXd S = MatrixXd::Zero(eigenvalues.rows(), eigenvalues.rows());
			for (int s = 0; s < eigenvalues.rows(); s++)
				S(s, s) = eigenvalues(s) > -1e-10 ? eigenvalues(s) : 0;
			K_stretch9 = -eigenvectors * S * eigenvectors.transpose();
		}

        int tripletOffset = f * 54;
        
		// create triplets - only for the lower triangle of the whole matrices K and D
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j <= i; j++) {	// go through all pairs of vertices
				for (int p = 0; p < 3; p++) {		// x, y, z
					for (int q = 0; q <= p; q++) {	// x, y, z
						int idi = F(f, i);			// we always want F(f,i) < F(f,j) here, such that we only fill the lower triangular matrix
						int idj = F(f, j);
						int idp = p;				// we might need to swap p and q, since we might need to use K_ji.transpose()
						int idq = q;				// K_ij is not symmetric here
						if (F(f, i) > F(f, j)) {
							idi = F(f, j);
							idj = F(f, i);
							idp = q;
							idq = p;
						}

                /*
						int row = idi + p * n; // corresponds to vertex id (i/j) and shifted by number of vertices for y and z coordinates (p)
						int col = idj + q * n;

                        triK.push_back(Tri(row, col, K_stretch9(3*i + idp, 3*j + idq)));            // ... these are always in the lower triangle
                        triD.push_back(Tri(row, col, D_stretch9(3*i + idp, 3*j + idq)));
                    */
                        
                        triK[tripletOffset] = Eigen::Triplet<double>(0, 0, K_stretch9(3*i + idp, 3*j + idq));
                        triD[tripletOffset] = Eigen::Triplet<double>(0, 0, D_stretch9(3*i + idp, 3*j + idq));
                        ++tripletOffset;
                        
					
						if (i != j) {
                

                            triK[tripletOffset] = Eigen::Triplet<double>(0, 0, K_stretch9(3*i + idq, 3*j + idp));
                            triD[tripletOffset] = Eigen::Triplet<double>(0, 0, D_stretch9(3*i + idq, 3*j + idp));
                            ++tripletOffset;
                            
                        /*
                            row = idj + p * n; 														// only i and j are switched
							col = idi + q * n;
							triK.push_back(Tri(row, col, K_stretch9(3*i + idq, 3*j + idp)));		// K_ij = K_ji.transpose() and symmetric!
							triD.push_back(Tri(row, col, D_stretch9(3*i + idq, 3*j + idp)));		// but D_ij is not symmetric, why we need to do it this whole complicated way
						*/
                         }
					}
				}
			}
		}
	}

	// build the sparse matrices from the triplets
    igl::sparse_cached(triK, K_data, this->K);
    igl::sparse_cached(triD, D_data, this->D);
}
