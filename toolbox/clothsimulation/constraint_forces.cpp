#include "constraint_forces.h"

#include <igl/sparse_cached.h>

using namespace std;
using namespace Eigen;

typedef Triplet<double> Tri;

Constraints::Constraints() {}

void Constraints::init(
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

void Constraints::precompute_rest_shape(std::vector<int> constrained_vert_id, vector<Vector3d> constrained_vert_target){
	this->constrained_vert_id = constrained_vert_id;
	this->constrained_vert_target = constrained_vert_target;
	k_redo = true;
}

void Constraints::compute_forces(
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
		int x = constrained_vert_id[i];
		Vector3d target = constrained_vert_target[i];
		Vector3d v;
		for (int j = 0; j < 3; j++)
			v(j) = V(x + j * n);

		// condition vector
		Vector3d diff = X.row(x).transpose() - target;
		double C = (diff).dot(diff);

		// first order derivatives
		Vector3d C_dv = 2 * diff;
		double   Cdot = C_dv.transpose() * v;
		Vector3d F_constraints = -k_constraints * (C_dv * C);
		Vector3d D_constraints = -k_damping * C_dv * Cdot;

		for (int j = 0; j < 3; j++)
			Force(x + j * n) += F_constraints(j) + D_constraints(j);

		// second order derivatives

		// compute stiffness elements
		// this is a diagonal matrix with the same values along the diagonal.
		// So we can just multiply with the double instead afterwards to save comp. time 
		double dC_dvdv = 2.;

		Matrix3d K_constraints = C_dv * C_dv.transpose();
		K_constraints.diagonal() += (dC_dvdv * C) * Vector3d::Ones();
		K_constraints *= -k_constraints;

		// compute damping
		Matrix3d KD = -k_damping * dC_dvdv * Cdot * Matrix3d::Identity();
		K_constraints += KD;

		// damping elements
		Matrix3d D_constraints_m = -k_damping * C_dv * C_dv.transpose();
		
		/*
		// project onto positive eigenvalues
		// these 9x9 matrices are real symmetric => diagonizable (spectral theorem)
		// symmetry also gives orthogonal eigenvalues, therefore Eval.transpose() = Eval.inverse()
		SelfAdjointEigenSolver<MatrixXd> eig(-K_shear9);
		VectorXd eigenvalues = eig.eigenvalues();
		if (eigenvalues.minCoeff() < -1e-10) {
			MatrixXd eigenvectors = eig.eigenvectors();
			MatrixXd S = MatrixXd::Zero(eigenvalues.rows(), eigenvalues.rows());
			for (int s = 0; s < eigenvalues.rows(); s++)
				S(s, s) = eigenvalues(s) > -1e-10 ? eigenvalues(s) : 0;
			K_shear9 = -eigenvectors * S * eigenvectors.transpose();
		}
*/
		// create triplets - only for the lower triangle of the whole matrices K and D
		for (int p = 0; p < 3; p++) {		// x, y, z
			for (int q = 0; q <= p; q++) {	// x, y, z
				int row = x + p * n; // corresponds to vertex id (i/j) and shifted by number of vertices for y and z coordinates (p)
				int col = x + q * n;

				triK.push_back(Tri(row, col, K_constraints(p, q)));			// ... these are always in the lower triangle
				triD.push_back(Tri(row, col, D_constraints_m(p, q)));
			}
		}
	}

	// build the sparse matrices from the triplets
	if (k_redo) {
		this->K = SparseMatrix<double>(n3, n3);
		this->D = SparseMatrix<double>(n3, n3);
		igl::sparse_cached_precompute(triK, K_data, this->K);
		igl::sparse_cached_precompute(triD, D_data, this->D);
		cout << "initialize matrix structure" << endl;
		K = this->K;
		D = this->D;
		k_redo = false;
	}
	else {
		igl::sparse_cached(triK, K_data, this->K);
		igl::sparse_cached(triD, D_data, this->D);
		K = this->K;
		D = this->D;
	}
}
