#include "bend_quadratic_forces.h"

#include <igl/sparse_cached.h>
#include <iostream>
#include <Eigen/Dense>
#include <igl/edges.h>
#include "../adjacency.h"
#include "../timer.h"

using namespace std;
using namespace Eigen;

typedef Eigen::Triplet<double> Tri;

// reference code for Discrete Quadratic Curvature Energies is here
// http://www.cs.columbia.edu/cg/quadratic/
// I took the functions cotTheta and Compute LocalStiffness and made them works with Eigen

//         x2
//         /\
//        /  \
//     e1/    \e3
//      /  t0  \
//     /        \
//    /    e0    \
//  x0------------x1
//    \          /
//     \   t1   /
//      \      /
//     e2\    /e4
//        \  /
//         \/
//         x3
//
// Edge orientation: e0,e1,e2 point away from x0
//                      e3,e4 point away from x1

Bend::Bend() {}

double Bend::cotTheta(const Vector3d v, const Vector3d w)
{
	//assert(finite(v.length()));
	//assert(finite(w.length()));
	//assert(v.length() > 0);
	//assert(w.length() > 0);
	const double cosTheta = v.dot(w);
	const double sinTheta = v.cross(w).norm();
	return (cosTheta / sinTheta);
}

// compute 4 by 4 local stiffness matrix Q(e0)
void Bend::ComputeLocalStiffness(
	const vector< Vector3d>& x,
	Matrix4d& Q)
{
	const Vector3d e0 = x[1] - x[0];
	const Vector3d e1 = x[2] - x[0];
	const Vector3d e2 = x[3] - x[0];
	const Vector3d e3 = x[2] - x[1];
	const Vector3d e4 = x[3] - x[1];

	const double c01 = cotTheta(e0, e1);
	const double c02 = cotTheta(e0, e2);
	const double c03 = cotTheta(-e0, e3);
	const double c04 = cotTheta(-e0, e4);

	const Vector4d K0 = Vector4d(c03 + c04, c01 + c02, -c01 - c03, -c02 - c04);

	const double A0 = e0.cross(e1).norm();
	const double A1 = e0.cross(e2).norm();

	const double coef = -3. / (A0 + A1);

	assert(finite(coef));
	assert(finite(c01));
	assert(finite(c02));
	assert(finite(c03));
	assert(finite(c04));

	// compute Q = coef times outer product of K0 and K0
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < i; ++j) {
			Q(i, j) = Q(j, i) = coef * K0[i] * K0[j];
		}
		Q(i, i) = coef * K0[i] * K0[i];
	}
}

void Bend::init(
	const double k_bend,
	const double k_damping,
	const MatrixXd& X,			// in: vertex positions
	const MatrixXi& T			// in: mesh triangles
) {
	this->k_bend = k_bend;
	this->k_damping = k_damping;
	this->n = X.rows();

	// create list of 4 vertices for each face pair (each internal edge)
	createFacePairEdgeListWith4VerticeIDs(T, E4);

	// precompute the matrices Q, K and D
	K.resize(3 * n, 3 * n);
	D.resize(n * 3, n * 3);
	Q.resize(E4.rows());

    F_local.resize(E4.rows() * 4);
    
	this->precompute_rest_shape(X);
}

void Bend::precompute_rest_shape(const MatrixXd& X){

    
    if(!tripletsInitialized)
    {
        std::vector<Eigen::Triplet<double>> tri;
        
        for (int e = 0; e < E4.rows(); e++)
        {
            // add stiffness matrix
            for (int dim = 0; dim < 3; dim++) {
                for (int v1 = 0; v1 < 4; v1++) {
                    for (int v2 = 0; v2 < v1; v2++) {
                        tri.push_back(Tri(E4(e, v1) + dim * n, E4(e, v2) + dim * n, 0));
                        tri.push_back(Tri(E4(e, v2) + dim * n, E4(e, v1) + dim * n, 0));
                    }
                    
                    tri.push_back(Tri(E4(e, v1) + dim * n, E4(e, v1) + dim * n, 0));
                }
            }
        }
        
        K = SparseMatrix<double>(3 * n, 3 * n);
        igl::sparse_cached_precompute(tri, K_data, K);
        D = K;
        
        triK.resize(tri.size());
        triD.resize(tri.size());
        
        tripletsInitialized = true;
    }
    
#pragma omp parallel for
    for (int e = 0; e < E4.rows(); e++) {
        vector< Vector3d > x(4);
        for (int i = 0; i < 4; i++)
            x[i] = X.row(E4(e, i));

        ComputeLocalStiffness(x, Q[e]);

        // --- second order derivatives ---
        // stiffness and damping matrix
        Matrix4d K_local = k_bend * Q[e];
        Matrix4d D_local = k_damping * Q[e];

        int offset = e * 48;
        
        // add stiffness matrix
        for (int dim = 0; dim < 3; dim++) {
            for (int v1 = 0; v1 < 4; v1++) {
                for (int v2 = 0; v2 < v1; v2++) {
                    
                    triK[offset] = Tri(0, 0, K_local(v1, v2));
                    triK[offset+1] = Tri(0, 0, K_local(v2, v1));
                    
                    triD[offset] = Tri(0, 0, D_local(v1, v2));
                    triD[offset+1] = Tri(0, 0, D_local(v2, v1));
                    
                    offset += 2;
                }
                
                
                triK[offset] = Tri(0, 0,  K_local(v1, v1));
                triD[offset] = Tri(0, 0,  D_local(v1, v1));
                
                offset++;
            }
        }
    }
    
    igl::sparse_cached(triK, K_data, this->K);
    igl::sparse_cached(triD, K_data, this->D);
}


const Eigen::SparseMatrix<double>& Bend::getK() const
{
    return K;
}

const Eigen::SparseMatrix<double>& Bend::getD() const
{
    return D;
}
  

void Bend::compute_forces(
	const MatrixXd& X,			// in: vertex positions
	const VectorXd& V,			// in: velocities
	VectorXd& F)				// out: forces
{
    //#pragma omp parallel for
	for (int e = 0; e < E4.rows(); e++) {
		
        // get necessary positions and velocities for this edge
		vector< Vector4d > x(3);				// position: 3 vectors: x_coords, y_coords and z_coords - each with 4 entries for 4 vertices
		vector< Vector4d > v(3);				// velocity
		
        for (int i = 0; i < 4; i++) {
			int vertex = E4(e, i);
            for (int j = 0; j < 3; j++) {
				x[j](i) = X(vertex, j);
				v[j](i) = V(vertex + j * n);
			}
		}

        const auto qe = Q[e];
        
        F_local[3 * e + 0] = qe * (k_bend * x[0] + k_damping * v[0]);
        F_local[3 * e + 1] = qe * (k_bend * x[1] + k_damping * v[1]);
        F_local[3 * e + 2] = qe * (k_bend * x[2] + k_damping * v[2]);
	}
            
    for (int e = 0; e < E4.rows(); e++) {
        for (int i = 0; i < 4; i++) {            // force on vertex i
            const int vertex = E4(e, i);
            for (int j = 0; j < 3; j++) {        // put each force dimension into the right place of the total F
                F(vertex + j * n) += F_local[3 * e + j](i);
            }
        }
    }
}
