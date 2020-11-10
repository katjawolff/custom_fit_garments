#include "rest_shape.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <igl/edges.h>
#include "../adjacency.h"
#include <igl/cotmatrix.h>
#include <igl/cotmatrix_entries.h>
#include <igl/slice_into.h>
#include "../timer.h"

#include <omp.h>

using namespace std;
using namespace Eigen;

typedef Triplet<double> Tri;


EvolveRestShape::EvolveRestShape(){}

void EvolveRestShape::init(const MatrixXd& V, const MatrixXi& F) {
    // create adjacency information - E4: list of 4 vertices for each face pair (each internal edge)
    this->F = F;
    igl::edges(F, E);
    vector< vector<int> > VF_adj;
    createVertexFaceAdjacencyList(F, VF_adj);
    createFaceEdgeAdjecencyList(F, E, VF_adj, FE_adj);
    createFacePairEdgeListWith4VerticeIDs(F, E, VF_adj, E4, EF6, EF_adj);
    faces = F.rows();
    edges = E.rows();
    verts = V.rows();
    
    
    Vinit = V;
    igl::cotmatrix(V, F, L);
    Eigen::SparseMatrix<double> A(verts, verts);
    A.setIdentity();
    A = lambda * A - L;
    
    solver2.setSystem(A);
    igl::cotmatrix_entries(V, F, cotanEntries);
    
    interior_edges = 0;
    for (int i = 0; i < EF_adj.rows(); i++)
        if (EF_adj(i, 1) != -1)
            interior_edges++;
}

void EvolveRestShape::setFixedVertices(const vector<bool>& fixed_boundary_vert) {
    this->fixed_boundary_vert = fixed_boundary_vert;

    fixed_boundary_edge.clear();
    fixed_boundary_edge.resize(E.rows(), false);
    for (int e = 0; e < E.rows(); e++) 
        if (fixed_boundary_vert[E(e, 0)] && fixed_boundary_vert[E(e, 1)])
            fixed_boundary_edge[e] = true;
}

void EvolveRestShape::energy(
    const MatrixXd& Xi,                     // current restshape vertex positions
    const MatrixXd& X0,                     // last restshape vertex positions
    const vector<double>& length_target,
    const vector<int>& edge_constraints,
    Eigen::VectorXd& f
) {
    double en = .0;
    
    f.resize(6 * verts + edges);
    f.setZero();
    
    // Laplace energy
    Eigen::Map<Eigen::MatrixXd>(f.data(), verts, 3) = alpha * (L * (Xi - X0));
    int cnt = 3 * verts;
    
    // Position energy
    std::vector<char> vertex_constrained_by_edge(verts, false);
    for(int i = 0; i < edges; ++i)
        if(edge_constraints[i])
        {
            vertex_constrained_by_edge[E(i, 0)] = true;
            vertex_constrained_by_edge[E(i, 1)] = true;
        }
    
    for(int i = 0; i < verts; ++i)
    {
        const double w0 = vertex_constrained_by_edge[i] ? eps : lambda + eps;
        Eigen::Vector3d w = w0 * (Xi.row(i) - X0.row(i));
                       
        f(cnt + i) = w(0);
        f(cnt + verts + i) = w(1);
        f(cnt + 2 *  verts + i) = w(2);
    }
    
    // Edge energy
    cnt += 3 * verts;
    for(int i = 0; i < edges; ++i)
    {
        if(!fixed_boundary_edge[i])
            f(cnt + i) = (Xi.row(E(i, 1)) - Xi.row(E(i, 0))).squaredNorm() - pow(length_target[i], 2);
    }
}



void EvolveRestShape::jacobianConstant(
    const MatrixXd& Xi,
    const vector< int >& edge_constraints)
{
    // Laplace Energy
    std::vector<Eigen::Triplet<double>> trip;
    double* lvals = L.valuePtr();
    
    for(int i = 0; i < verts; ++i)
    {
        for(int j = L.outerIndexPtr()[i]; j < L.outerIndexPtr()[i+1]; ++j)
        {
            int rj = L.innerIndexPtr()[j];
            const double w = alpha * lvals[j];
            trip.emplace_back(i, rj, w);
            trip.emplace_back(verts + i, verts + rj, w);
            trip.emplace_back(2 * verts + i, 2 * verts + rj, w);
        }
    }
    
    int offset = 3 * verts;
        
    // Position energy
    std::vector<char> vertex_constrained_by_edge(verts, false);
    for(int i = 0; i < edges; ++i)
        if(edge_constraints[i])
        {
            vertex_constrained_by_edge[E(i, 0)] = true;
            vertex_constrained_by_edge[E(i, 1)] = true;
        }
    
    int vfree = 0;
    
    for(int i = 0; i < verts; ++i)
        if(!vertex_constrained_by_edge[i]) ++vfree;

    
    int cnt = 0;

    
    for(int i = 0; i < verts; ++i)
    {
        const double w = vertex_constrained_by_edge[i] ? eps : lambda + eps;
     
        trip.emplace_back(offset + i, i, w);
        trip.emplace_back(offset + verts + i, verts + i, w);
        trip.emplace_back(offset + 2 * verts + i, 2 * verts + i, w);
    }
    
    offset += 3 * verts;
       
    J0.resize(6 * verts , 3 * verts);
    J0.setFromTriplets(trip.begin(), trip.end());
    JTJ0 = J0.transpose() * J0;
}

void EvolveRestShape:: jacobianUpdate(
    const Eigen::MatrixXd& Xi,
    const std::vector< int >& edge_constraints,
    Eigen::SparseMatrix<double>& J
)
{
    std::vector<Eigen::Triplet<double>> trip;
    
    for(int i = 0; i < edges; ++i)
    {
        if(!fixed_boundary_edge[i])
        {
            const Eigen::Vector3d e = Xi.row(E(i, 1)) - Xi.row(E(i, 0));
            
            for(int j = 0; j < 3; ++j)
            {
                trip.emplace_back(i, j * verts + E(i, 1),  2 * e(j));
                trip.emplace_back(i, j * verts + E(i, 0), -2 * e(j));
            }
        }
    }
    
    J.resize(edges, 3 * verts);
    J.setFromTriplets(trip.begin(), trip.end());
}

double EvolveRestShape::checkFiniteDifferences(
    const MatrixXd& Xi,
    const MatrixXd& X0,
    const vector< double >& length_target,
    const vector< int >& edge_constraints
){
   // igl::cotmatrix(X0, F, L);
    

    const double eps = 1e-6;
    MatrixXd X = Xi;
    Eigen::SparseMatrix<double> J;
    VectorXd f, f0;
   
    Eigen::VectorXd mc2 = (L * X0).rowwise().squaredNorm();
    
    energy(X, X0, length_target, edge_constraints, f0);
    //jacobian(X, edge_constraints, J);
    
    Eigen::MatrixXd Jdense(J.rows(), J.cols());
    
    for(int i = 0; i < verts; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            X(i, j) += eps;
            
            energy(X, X0, length_target, edge_constraints, f);
            Jdense.col(j * verts + i) = (f - f0) / eps;
            
            X(i, j) -= eps;
        }
    }
    
    Eigen::MatrixXd J2 = J.toDense();
    Eigen::MatrixXd diff = (J2 - Jdense);
    
    
    double maxVal = .0;
    double maxVal2 = 0;
    double maxVal3 = .0;
    
    for(int j = 0; j < J.cols(); ++j)
    for(int i = 0; i < J.rows(); ++i)
        if(std::abs(J2(i,j)) > 1e-2 )
        {
            double w = abs((J2(i, j) - Jdense(i, j)) / J2(i, j));
            if( w > maxVal)
            {
                maxVal = w;
                maxVal2 = J2(i, j);
                maxVal3 = (J2(i, j) - Jdense(i, j));
            }
        }
    
    std::cout << "max fd error: " << maxVal << " " << maxVal3 << " " << maxVal2 << "\n";
    
 //   Eigen::saveMarket(J, "../J.mtx");
   // Eigen::saveMarket(Jdense, "../J2.mtx");
}

void EvolveRestShape::adjust_restshape(
    const MatrixXd& X_simulation,
    const MatrixXd& Vm,
    const double t_stretch,
    const double t_compress,
    MatrixXd& X_restshape
) {
    vector<double> edge_adjustment(edges, 0.);
    adjust_restshape(X_simulation, Vm, t_stretch, t_compress, edge_adjustment, X_restshape);
}

void trianglesTo2d(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<Eigen::Matrix2d>& tris)
{
    tris.resize(F.rows());
    
    for(int i = 0; i < F.rows(); ++i)
    {
        Eigen::Vector3d e0 = V.row(F(i, 1)) - V.row(F(i, 0));
        Eigen::Vector3d e1 = V.row(F(i, 2)) - V.row(F(i, 0));
        Eigen::Vector3d n = e0.cross(e1).normalized();
            
        double l0 = e0.norm();
        Eigen::Vector3d e0n = e0 / l0;
        Eigen::Vector3d e01 = n.cross(e0n);
        
        Eigen::Matrix2d t;
        t << l0, e1.dot(e0n), .0, e1.dot(e01);
        
        tris[i] = t;
    }
}

void EvolveRestShape::arap(std::vector<Eigen::Matrix2d>& X0, MatrixXd& Xi, int iters)
{
    // local step
    Eigen::MatrixXd b(Xi.rows(), 3);

    Eigen::DiagonalMatrix<double, 3> Iflip(3);
    Iflip.setIdentity();
    Iflip.diagonal()(2) = -1;
    

    // compute frames for X0 and invert
    std::vector<Eigen::Matrix3d> invFrame0(F.rows());
    std::vector<std::array<Eigen::Vector3d, 3>> triangleEdges(F.rows());
    std::vector<std::array<Eigen::Vector3d, 3>> bvalues(F.rows());
    
    for(int i = 0; i < F.rows(); ++i)
    {
        Eigen::Matrix3d t0;
        t0.setZero();
        t0(1, 0) = X0[i](0,0);
        t0(1, 1) = X0[i](1,0);
        
        t0(2, 0) = X0[i](0,1);
        t0(2, 1) = X0[i](1,1);
    
        for(int k = 0; k < 3; ++k)
            triangleEdges[i][k] = t0.row((k+2) % 3) - t0.row((k+1) % 3);
        
        // extra point
        Eigen::Vector3d x03 = t0.colwise().mean() + (t0.row(1) - t0.row(0)).cross(t0.row(2) - t0.row(0));
        
        Eigen::Matrix3d frame0;
        for(int k = 0; k < 3; ++k)
            frame0.col(k) = t0.row(k).transpose() - x03;
        
        invFrame0[i] = frame0.inverse();
    }
        
    
    for(int k = 0; k < iters; ++k)
    {

#pragma omp parallel for
        for(int i = 0; i < F.rows(); ++i)
        {
            // compute current frame
            Eigen::Matrix3d t;
            for(int k = 0; k < 3; ++k)
                t.row(k) = Xi.row(F(i, k));
            
            // extra point
            Eigen::Vector3d x3 = t.colwise().mean() + (t.row(1) - t.row(0)).cross(t.row(2) - t.row(0));
            
            Eigen::Matrix3d framei;
            for(int k = 0; k < 3; ++k)
                framei.col(k) = t.row(k).transpose() - x3;
            
            // frame mapping
            Eigen::Matrix3d T = framei * invFrame0[i];
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(T, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
            Eigen::Matrix3d R;
            
            if((svd.matrixU() * svd.matrixV().transpose()).determinant() < .0)
            {
                R = svd.matrixU() * Iflip * svd.matrixV().transpose();
            } else
            {
                R = svd.matrixU()* svd.matrixV().transpose();
            }

            for(int k = 0; k < 3; ++k)
            {
                Eigen::Vector3d bk = R * triangleEdges[i][k];
                bvalues[i][k] = cotanEntries(i, k) * bk;
            }
        }
        
        b.setZero();
        for(int i = 0; i < F.rows(); ++i)
        {
            for(int k = 0; k < 3; ++k)
            {
                b.row(F(i, (k+2) % 3)) += bvalues[i][k];
                b.row(F(i, (k+1) % 3)) -= bvalues[i][k];
            }
        }
        
        b += lambda * Xi;
        
        // global step
        Eigen::MatrixXd Xiold = Xi;
        solver2.solve(b, Xi);
        
        if( (Xi - Xiold).norm() < 1e-8 )
        {
            std::cout << "arap: converged in " << k << " iterations.\n";
            return;
        }
    }
    

    std::cout << "arap: iteration limit exceeded.\n";
}

void EvolveRestShape::adjust_restshape(
    const MatrixXd& X_simulation,
    const MatrixXd& Vm,
    const double t_stretch,
    const double t_compress,
    const vector<double>& edge_adjustment,
    MatrixXd & X_restshape
) {
    verts = X_restshape.rows();
    
    vector< double > l_target(edges);
    double original_diff = 0;
    int edge_constraints_count = 0;
    vector<int> edge_constraints(edges, 0);
    
    // compute per triangle adjustment factor
    // -------------------------------------
    vector<double> faceScaleFactor(F.rows());
    
    for(int i = 0; i < F.rows(); ++i)
    {
        double f = .0;
        
        for(int j = 0; j < 3; ++j)
        {
            f += edge_adjustment[FE_adj[i][j]];
        }
        
        faceScaleFactor[i] = f / 3.;
    }
    
    // adjust rest shape triangles
    // -------------------------------------
    std::vector<Eigen::Matrix2d> triSim, triRest;
    trianglesTo2d(X_simulation, F, triSim);
    trianglesTo2d(X_restshape, F, triRest);

    double tmin = t_compress;
    double tmax = t_stretch;
        
    auto stretchError = [&](const Eigen::MatrixXd& V){
        std::vector<Eigen::Matrix2d> tris;
        trianglesTo2d(V, F, tris);
            
        double err = .0;
            
        for(int i = 0; i < F.rows(); ++i){
            Eigen::Matrix2d f = triSim[i] * tris[i].inverse();
            Eigen::JacobiSVD<Eigen::Matrix2d> svd(f);
                
            for(int k = 0; k < 2; ++k){
                double s = svd.singularValues()(k);
                    
                if( s > tmax) err += (s - tmax) * (s - tmax);
                if( s < tmin) err += (s - tmin) * (s - tmin);
            }
        }

        return err;
    };

    for(int i = 0; i < F.rows(); ++i){
        Eigen::Matrix2d f = triRest[i] * triSim[i].inverse();
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(f, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d singularValues = svd.singularValues() * faceScaleFactor[i];
        bool adapted = false;
            
        // adapt singular values
        for(int j = 0; j < 2; ++j){
            double& s = singularValues(j);
                
            if(1. / (s * s) < tmin){
                s = 1. / sqrt(tmin);
                adapted = true;
            } else if(1. / (s * s) > tmax){
                s = 1. / sqrt(tmax);
                adapted = true;
            }
        }
            
        //if(adapted){
            Eigen::Matrix2d f2 = svd.matrixU() * Eigen::DiagonalMatrix<double, 2>(singularValues) * svd.matrixV().transpose();
            triRest[i] = f2 * triSim[i];
                
            for(int j = 0; j < 3; ++j){
                if(!edge_constraints[FE_adj[i][j]])
                    edge_constraints[FE_adj[i][j]] = ++edge_constraints_count;
            }
        //}
            /*
        if(l_target[FE_adj[i][0]]){
            l_target[FE_adj[i][0]] += (triRest[i].col(0) - triRest[i].col(1)).norm();
            l_target[FE_adj[i][0]] *= 0.5;
        } else
            l_target[FE_adj[i][0]] = (triRest[i].col(0) - triRest[i].col(1)).norm();
            
        if(l_target[FE_adj[i][1]]){
            l_target[FE_adj[i][1]] +=  triRest[i].col(1).norm();
            l_target[FE_adj[i][1]] *= 0.5;
        } else
            l_target[FE_adj[i][1]] = triRest[i].col(1).norm();
            
        if(l_target[FE_adj[i][2]]){
            l_target[FE_adj[i][2]] +=  triRest[i].col(0).norm();
            l_target[FE_adj[i][2]] *= 0.5;
        } else
            l_target[FE_adj[i][2]] = triRest[i].col(0).norm();  */
    }
  
        
    if (edge_constraints_count== 0){
        std::cout << "all edges ok\n";
        return;
    } else
        std::cout << "adapting restshape\n";
        
    // evaluation of stretch error is somewhat costly and should be disabled for time measurements
        
    // double err0 = stretchError(X_restshape) ;
    arap(triRest, X_restshape, 25);
    //std::cout << "error before: " << err0 << " error after: " << stretchError(X_restshape) << std::endl;
}




