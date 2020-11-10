#include "seam_forces.h"

#include <igl/sparse_cached.h>

using namespace std;
using namespace Eigen;

typedef Triplet<double> Tri;

Seams::Seams() {}

void Seams::init(
	double k_constraints, 
	double k_damping, 
	int n
) {
	this->k_constraints = k_constraints;
	this->k_damping = k_damping;

	this->n = n;
	n3 = 3 * n;
	c = 9 * constrained_vert_id.size();
}

void Seams::precompute_rest_shape(vector< pair<int, int> > constrained_vert_id){
	this->constrained_vert_id = constrained_vert_id;
}

void Seams::compute_forces(
	const MatrixXd& X,		// in: vertex positions
	const VectorXd& V,		// in: velocitiy at vertex positions
	VectorXd& Force,		// out: force vector
	SparseMatrix<double>& K,// out: stiffness matrix
	SparseMatrix<double>& D	// out: damping matrix
) {
	vector<Tri> triK, triD;
	triK.reserve(c);
	triD.reserve(c);

	// only do calculations if k != 0
	if (constrained_vert_id.size() == 0) {
		K = SparseMatrix<double>(n3, n3);	// zero matrices
		D = SparseMatrix<double>(n3, n3);
		return;
	}

	for (int i = 0; i < constrained_vert_id.size(); i++) {
		int x1 = constrained_vert_id[i].first;
        int x2 = constrained_vert_id[i].second;
		Vector3d v1, v2;
		for (int j = 0; j < 3; j++) {
            v1(j) = V(x1 + j * n);
            v2(j) = V(x2 + j * n);
        }

		// condition vector
		Vector3d diff = X.row(x1).transpose() - X.row(x2).transpose();
		double C = (diff).dot(diff);

		// first order derivatives
		Vector3d C_dv1 = 2 * diff;
        Vector3d C_dv2 = - 2 * diff;
		double   Cdot_v1 = C_dv1.transpose() * v1;
        double   Cdot_v2 = C_dv2.transpose() * v2;
		Vector3d F_constraints_x1 = -k_constraints * (C_dv1 * C);
        Vector3d F_constraints_x2 = -k_constraints * (C_dv2 * C);
		Vector3d D_constraints_x1 = -k_damping * C_dv1 * Cdot_v1;
        Vector3d D_constraints_x2 = -k_damping * C_dv2 * Cdot_v2;

		for (int j = 0; j < 3; j++) {
            Force(x1 + j * n) += F_constraints_x1(j) + D_constraints_x1(j);
            Force(x2 + j * n) += F_constraints_x2(j) + D_constraints_x2(j);
        }

		// second order derivatives

		// compute stiffness elements
		// this is a diagonal matrix with the same values along the diagonal.
		// So we can just multiply with the double instead afterwards to save comp. time 
		double dC_dvdv = 2.;

		Matrix3d K_constraints_x1 = C_dv1 * C_dv1.transpose();             // same for both x1, x2, since the sign cancels out
		K_constraints_x1.diagonal() += (dC_dvdv * C) * Vector3d::Ones();   // but then here we have (-) for x2
		K_constraints_x1 *= -k_constraints;
		Matrix3d K_constraints_x2 = - K_constraints_x1;

		// compute damping
		Matrix3d KD = -k_damping * dC_dvdv * Cdot_v1 * Matrix3d::Identity(); // sign cancels out
		K_constraints_x1 += KD;
        K_constraints_x2 += KD;

		// damping elements
		Matrix3d D_constraints_m = -k_damping * C_dv1 * C_dv1.transpose();  // sign cancels out

		// create triplets - only for the lower triangle of the whole matrices K and D
		for (int p = 0; p < 3; p++) {		// x, y, z
			for (int q = 0; q <= p; q++) {	// x, y, z
				int row1 = x1 + p * n; // corresponds to vertex id (i/j) and shifted by number of vertices for y and z coordinates (p)
				int col1 = x1 + q * n;
				triK.push_back(Tri(row1, col1, K_constraints_x1(p, q)));			// ... these are always in the lower triangle
				triD.push_back(Tri(row1, col1, D_constraints_m(p, q)));

                int row2 = x2 + p * n; // corresponds to vertex id (i/j) and shifted by number of vertices for y and z coordinates (p)
                int col2 = x2 + q * n;
                triK.push_back(Tri(row2, col2, K_constraints_x1(p, q)));			// ... these are always in the lower triangle
                triD.push_back(Tri(row2, col2, D_constraints_m(p, q)));
			}
		}
	}

	// build the sparse matrices from the triplets
	if (this->K.rows() == 0) {
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
}
