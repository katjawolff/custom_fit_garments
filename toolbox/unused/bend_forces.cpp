#include "bend_forces.h"

#include "../adjacency.h"

#include <Eigen/Dense>
#include <igl/edges.h>
#include <igl/sparse_cached.h>
#include <iostream>
#include <iostream>
#include <cmath>
#include <cfloat>

using namespace std;
using namespace Eigen;

typedef Eigen::Triplet<double> Tri; 

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

// Refer to: Discrete bending forces and their Jacobians by Tamstorf et al.
//
//         x0
//         /\
//        /  \
//     e2/    \e1
//      /  t   \
//     /        \
//    /    e0    \
//  x1------------x2		
//    \          /
//     \   t~   /
//      \      /
//    e2~\    /e1~
//        \  /
//         \/
//         x3
//
// Edge orientation: e0,e2,e2~ point away from x1
//                      e1,e1~ point away from x2
// E4 keeps them in the following order: x1, x2, x0, x3

Bend::Bend() {}

void Bend::init(
	const double k_bend,
	const double k_damping,
	const MatrixXd& X,			// in: vertex positions
	const MatrixXi& T			// in: mesh triangles
) {
	this->k_bend = k_bend;
	this->k_damping = k_damping;
	this->T = T;
	this->n = X.rows();
	this->m = T.rows();

	// create adjacency information - E4: list of 4 vertices for each face pair (each internal edge)
	vector< vector<int> > VF_adj;
	igl::edges(T, E);
	createVertexFaceAdjacencyList(T, VF_adj);
	createFaceEdgeAdjecencyList(T, E, VF_adj, FE_adj);
	createFacePairEdgeListWith4VerticeIDs(T, E, VF_adj, E4, EF6, EF_adj);
	this->no_edges = E4.rows();

	// define size of output
	n3 = 3 * n;
	k_entries = no_edges * 16 * 9;

	// precompute phi and scaling coefficient
	phi_bar.resize(no_edges);
	phi_hat.resize(no_edges);
	a.resize(no_edges);
	precompute_rest_shape(X);
}

void Bend::precompute_rest_shape(const MatrixXd & X) {
	for (int e = 0; e < no_edges; e++) {
		if (EF_adj(e, 1) != -1) { // no boundary edge
			// edges
			Vector3d e0 = X.row(E4(e, 2)) - X.row(E4(e, 1));
			Vector3d e2 = X.row(E4(e, 0)) - X.row(E4(e, 1));
			Vector3d e2_dot = X.row(E4(e, 3)) - X.row(E4(e, 1));

			// normals
			Vector3d normal = e0.cross(e2).normalized();
			Vector3d normal_dot = e2_dot.cross(e0).normalized();

			// phi of theta/2 of reference configuration
			Matrix3d det;
			det.col(0) = normal;
			det.col(1) = normal_dot;
			det.col(2) = e0;
			phi_bar(e) = 2. * sgn(det.determinant()) * (normal - normal_dot).norm() / (normal + normal_dot).norm();
			phi_hat(e) = phi_bar(e);

			//area and scaling coefficient
			double a0 = 0.5 * e0.cross(e2).norm();
			double a1 = 0.5 * e0.cross(e2_dot).norm();
			double length = e0.norm();
			a(e) = 3. * length * length / (a0 + a1);
		}
		else {
			phi_bar(e) = 0;
			phi_hat(e) = 0;
			a(e) = 0;
		}
	}
}

void Bend::compute_forces(
	const MatrixXd& X,			// in: vertex positions
	const double timestep,		// in: delta t
	VectorXd& F,				// out: forces
	SparseMatrix<double>& K,	// out: stiffness matrix
	SparseMatrix<double>& D		// out: damping matrix
) {
	F = VectorXd::Zero(n3);
	vector<Tri> triK, triD;
	triK.reserve(k_entries);
	triD.reserve(k_entries);

	// only do calculations if k_bend != 0
	if (k_bend == 0) {
		K = SparseMatrix<double>(n3, n3);	// zero matrices
		D = SparseMatrix<double>(n3, n3);
		return; 
	}

	// compute
	MatrixXd normal(m, 3);
	for (int f = 0; f < m; f++) {
		Vector3d e1 = X.row(T(f, 1)) - X.row(T(f, 0));
		Vector3d e2 = X.row(T(f, 2)) - X.row(T(f, 0));
		normal.row(f) = e1.cross(e2).normalized();
	}

	VectorXd edge_length(no_edges);
	VectorXd phi_d(no_edges);
	VectorXd psi_d(no_edges);
	VectorXd psi_dd(no_edges);
	for (int e = 0; e < no_edges; e++) {
		// hinge edge
		Vector3d e0 = X.row(E4(e, 2)) - X.row(E4(e, 1));
		edge_length(e) = e0.norm();

		if (EF_adj(e, 1) != -1) { // no border edge
			Vector3d n1 = normal.row(EF_adj(e, 0));
			Vector3d n2 = normal.row(EF_adj(e, 1));

			Matrix3d det;
			det.col(0) = n1;
			det.col(1) = n2;
			det.col(2) = e0;

			double n1_plus_n2_norm = (n1 + n2).norm();
			double sec_theta_half = 2. / n1_plus_n2_norm;
			double phi = 2. * sgn(det.determinant()) * (n1 - n2).norm() / n1_plus_n2_norm;
			phi_d(e) = sec_theta_half * sec_theta_half;
			double phi_dd = 0.5 * phi * sec_theta_half;

			psi_d(e) = 2. * a(e) * ( k_bend * phi_d(e) * (phi - phi_bar(e))				// force
							    + k_damping * phi_d(e) * (phi - phi_hat(e)) / timestep );	// damping

			psi_dd(e) = 2. * a(e) * ( k_bend * (phi_dd * (phi - phi_bar(e)) + phi_d(e) * phi_d(e))				// force
								 + k_damping * (phi_dd * (phi - phi_hat(e)) + phi_d(e) * phi_d(e)) / timestep);	// damping

			// update last phi
			phi_hat(e) = phi;
		} 
		else {
			psi_d(e) = 0;
			psi_dd(e) = 0;
		}
	}
	
	MatrixXd cos_alpha(m,3), h_inverse(m,3);
	vector<MatrixXd> H_triangle(m);
	for (int f = 0; f < m; f++) {
		vector<Vector3d> e(3);
		e[0] = X.row(T(f, 2)) - X.row(T(f, 1));		// edge opposite of v_0
		e[1] = X.row(T(f, 0)) - X.row(T(f, 2));		// opposite of v_1
		e[2] = X.row(T(f, 1)) - X.row(T(f, 0));		// opposite of v_2

		double area = 0.5 * e[0].cross(-e[2]).norm();

		vector<Vector3d> en(3);
		for (int i = 0; i < 3; i++)
			en[i] = e[i].normalized();			// direction is important, therefore we do not precompute them per edge earlier

		vector<Vector3d> edge_normal(3);
		for (int i = 0; i < 3; i++) {
			cos_alpha(f, i) = -en[(i + 1) % 3].dot(en[(i + 2) % 3]);	// angle at v_i
			h_inverse(f, i) = 0.5 * edge_length(FE_adj[f][i]) / area;	// height ending in v_i
			Vector3d nt = normal.row(f).transpose();
			edge_normal[i] = en[i].cross(nt);							// edge normal to edge opposite to v_i
		}
		
		vector<double> c(3);
		vector<Matrix3d> M(3), N(3), R(3);
		for (int i = 0; i < 3; i++) {
			int e_id = FE_adj[f][i];
			M[i] = normal.row(f).transpose() * edge_normal[i].transpose();
			N[i] = M[i] / (edge_length(e_id) * edge_length(e_id));

			bool edge_on_boundary = EF_adj(e_id, 1) == -1;
			double sigma = edge_on_boundary ? 0. : 1.;
			c[i] = sigma * psi_d(e_id);
			R[i] = c[i] * N[i];
		}

		vector<double> d(3);
		for (int i = 0; i < 3; i++) {
			int i_minus = (i + 2) % 3;
			int i_plus = (i + 1) % 3;
			d[i] = c[i_minus] * cos_alpha(f, i_plus) + c[i_plus] * cos_alpha(f, i_minus) - c[i];
		}

		// calculate the stiffness matrix contribution for a single triangle
		H_triangle[f].resize(9, 9);
		for (int i = 0; i < 3; i++) {
			int j = (i + 1) % 3;
			int k = (i + 2) % 3;

			double omega_ii = h_inverse(f, i) * h_inverse(f, i);
			Matrix3d H_ii = omega_ii * d[i] * (M[i].transpose() + M[i]) - R[j] - R[k];
			H_triangle[f].block(3*i, 3*i, 3, 3) = H_ii;

			// check if global orientation of edge matches with local ccw orientation -> only for matching orientations do we transpose R
			//
			//		   e0
			//   v2 -------- v1			Of e0 (i = 0) the first vertex in ccw order is
			//    \		    /			v1 = F(f, (i+1)%3)
			//     \   f   /
			//	e1  \     /  e2
			//       \	 /
			//		  \ /
			//        v0
			//
			int e_id = FE_adj[f][i];
			int first_vertex_local = T(f, j);
			int first_vertex_global = E(e_id, 0);
			Matrix3d R_local = R[k];
			if (first_vertex_local == first_vertex_global)
				R_local.transposeInPlace();

			double omega_ij = h_inverse(f, i) * h_inverse(f, j);
			Matrix3d H_ij = omega_ij * (d[i] * M[j].transpose() + d[j] * M[i]) + R_local;
			H_triangle[f].block(3 * i, 3 * j, 3, 3) = H_ij;
			H_triangle[f].block(3 * j, 3 * i, 3, 3) = H_ij.transpose();
		}
	}

	for (int e = 0; e < no_edges; e++) {
		if (EF_adj(e, 1) != -1) { // no border edge

			// indexing
			int f = EF_adj(e, 0);		// adjacent face 1
			int f_dot = EF_adj(e, 1);	// adjacent face 2

			vector<int> f_v(3), f_dot_v(3);	// f_v 0,1,2 corresponds to v0,v1,v2 --- f_dot_v 0,1,2 corresponds to v1,v2,v3
			for (int i = 0; i < 3; i++) {
				f_v[i] = EF6(e, i);
				f_dot_v[i] = EF6(e, 3 + i);
			}

			// first derivative 
			VectorXd delta_theta(12);
			delta_theta.segment(0, 3) = -h_inverse(f, f_v[0]) * normal.row(f);
			delta_theta.segment(3, 3) = cos_alpha(f, f_v[2]) * h_inverse(f, f_v[1]) * normal.row(f) + cos_alpha(f_dot, f_dot_v[1]) * h_inverse(f_dot, f_dot_v[0]) * normal.row(f_dot);
			delta_theta.segment(6, 3) = cos_alpha(f, f_v[1]) * h_inverse(f, f_v[2]) * normal.row(f) + cos_alpha(f_dot, f_dot_v[0]) * h_inverse(f_dot, f_dot_v[1]) * normal.row(f_dot);
			delta_theta.segment(9, 3) = -h_inverse(f_dot, f_dot_v[2]) * normal.row(f_dot);

			for (int i = 0; i < 3; i++) 
				for (int v = 0; v < 4; v++) 
					F(E4(e, v) + i * n) += -psi_d(e) * delta_theta(3*v + i);

			// second derivative
			// stiffness matrix
			MatrixXd delta_f = MatrixXd::Zero(12, 12);
			delta_f -= psi_dd(e) * delta_theta * delta_theta.transpose();	// second of two terms that build delta_f

			for (int i = 0; i < 4; i++) {
				for (int j = 0; j <= i; j++) {	// go through vertice pairs i-j

					for (int p = 0; p < 3; p++) {			// go through x,y,z coordinates
						int q = 0;
						if (i == j) q = p;					// for central blocks, exploit symmetry
						for (; q < 3; q++) {
							if (i < 3 && j < 3) delta_f(3 * i + p, 3 * j + q) -= H_triangle[f](f_v[i] * 3 + p, f_v[j] * 3 + q);					// contribution of first term of face f
							if (i > 0 && j > 0) delta_f(3 * i + p, 3 * j + q) -= H_triangle[f_dot](f_dot_v[i - 1] * 3 + p, f_dot_v[j - 1] * 3 + q);	// contribution of first term of face f_dot
						}
					}
				}
			}

			// project onto positive eigenvalues
			// these 9x9 matrices are real symmetric => diagonizable (spectral theorem)
			// symmetry also gives orthogonal eigenvalues, therefore Eval.transpose() = Eval.inverse()
			SelfAdjointEigenSolver<MatrixXd> eig(-delta_f);
			VectorXd eigenvalues = eig.eigenvalues();
			if (eigenvalues.minCoeff() < -1e-10) {
				MatrixXd eigenvectors = eig.eigenvectors();
				MatrixXd S = MatrixXd::Zero(eigenvalues.rows(), eigenvalues.rows());
				for (int s = 0; s < eigenvalues.rows(); s++)
					S(s, s) = eigenvalues(s) > -1e-10 ? eigenvalues(s) : 0;
				delta_f = -eigenvectors * S * eigenvectors.transpose();
			}

			// create triplets
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j <= i; j++) {	// go through vertice pairs i-j

					for (int p = 0; p < 3; p++) {			// go through x,y,z coordinates
						int q = 0;
						if (i == j) q = p;					// for central blocks, exploit symmetry
						for (; q < 3; q++) {
							int row = E4(e, i) + p * n;		// corresponds to vertex id (i) and shifted by number of vertices for y and z coordinates (p)
							int col = E4(e, j) + q * n;
							if(col <= row) triK.push_back(Tri(row, col, delta_f(3 * i + p, 3 * j + q)));
							else triK.push_back(Tri(col, row, delta_f(3 * i + p, 3 * j + q)));
							//if((i != j) || (i == j && p != q)) triK.push_back(Tri(col, row, value));					// exploit symmetry of delta_f blocks
						}
					}
				}
			}

			// damping matrix
			// this is a hand-made damping matrix, that is not mentioned in the paper. Allows for bigger time steps.
			VectorXd D_diagonal = -2. * a(f) * k_damping * phi_d(e) * delta_theta;
			for (int i = 0; i < 3; i++){
				for (int v = 0; v < 4; v++) {
					int row = E4(e, v) + i * n;
					triD.push_back(Tri(row, row, D_diagonal(3 * v + i)));
				}
			}
		}
	}


	// build the sparse matrices from the triplets
	if (K_data.rows() == 0) {
		this->K = SparseMatrix<double>(n3, n3);
		this->D = SparseMatrix<double>(n3, n3);
		igl::sparse_cached_precompute(triK, K_data, this->K);
		igl::sparse_cached_precompute(triD, D_data, this->D);
		cout << "initialize matrix structure" << endl;
		K = this->K;
		D = this->D;
	}
	else {
		igl::sparse_cached(triK, K_data, this->K);
		igl::sparse_cached(triD, D_data, this->D);
		K = this->K;
		D = this->D;
	}

	//K.setFromTriplets(triK.begin(), triK.end());
	//D.setFromTriplets(triD.begin(), triD.end());
}
