#include "shear_forces.h"

#include <igl/sparse_cached.h>

using namespace std;
using namespace Eigen;

typedef Triplet<double> Tri;

Shear::Shear() {}

void Shear::init(double k_shear, double k_damping, MatrixXd& V, MatrixXi& F) {
	this->k_shear = k_shear;
	this->k_damping = k_damping;
	this->F = F;

	n = V.rows();
	m = F.rows();
	n3 = 3 * n;
	m81 = 81 * m; 	// pro f: 9 matrices * 9 entries

	precompute_rest_shape(V);
}

void Shear::precompute_rest_shape(MatrixXd& V) {
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
		Matrix2d Vref; Vref << v1.norm(), v2.dot(u), 0, v2.dot(w);
		V_inverse[i] = Vref.inverse();

		// some mysterious needed precomp (for dJ/dv_x), left col for J_x, right col for J_y
		B[i].resize(3, 2);
		B[i](0, 0) = -V_inverse[i](0, 0) - V_inverse[i](1, 0);
		B[i](0, 1) = -V_inverse[i](0, 1) - V_inverse[i](1, 1);
		B[i].block(1, 0, 2, 2) = V_inverse[i];
	}
}

void Shear::compute_forces(
	const MatrixXd& X,		// in: vertex positions
	const VectorXd& V,		// in: velocitiy at vertex positions
	VectorXd& Force,		// out: force vector
	SparseMatrix<double>& K,// out: stiffness matrix
	SparseMatrix<double>& D	// out: damping matrix
) {
	Force = VectorXd::Zero(n3);
	vector<Tri> triK, triD;
	triK.reserve(m81);
	triD.reserve(m81);

	// only do calculations if k != 0
	if (k_shear == 0) {
		K = SparseMatrix<double>(n3, n3);	// zero matrices
		D = SparseMatrix<double>(n3, n3);
		return;
	}

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

		// adjust for different mesh resolutions
		double k_shear_f = k_shear / a(f);

		// condition vector
		// from [Large Steps in Cloth Simulation, Baraff & Witkin, 1998] - Section 4.3
		double C_shear = a(f) * J.col(0).transpose() * J.col(1);

		// first order derivatives
		vector<Vector3d> C_dv(3);
		double Cdot = 0;			// derivative times velocity
		for (int i = 0; i < 3; i++) {			
			C_dv[i] = a(f) * (B[f](i, 0) * J.col(1) + B[f](i, 1) * J.col(0));			
			Cdot += C_dv[i].transpose() * v[i];
		}
		for (int i = 0; i < 3; i++) {
			// compute forces
			Vector3d F_shear = -k_shear_f * (C_dv[i] * C_shear);

			// compute damping
			Vector3d D_shear = -k_damping * C_dv[i] * Cdot;

			// force + damping - put into the right position of the global force vector
			for (int j = 0; j < 3; j++)
				Force(F(f, i) + j * n) += F_shear(j) + D_shear(j);
		}

		// second order derivatives
		MatrixXd K_shear9(9, 9);
		MatrixXd D_shear9(9, 9);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j <= i; j++) {	// go through all pairs of vertices
				// compute stiffness elements
				// this is a diagonal matrix with the same values along the diagonal.
				// So we can just multiply with the double instead afterwards to save comp. time 
				double dC_dvdv = a(f) * (B[f](i, 0) * B[f](j, 1) + B[f](i, 1) * B[f](j, 0));	
				
				Matrix3d K_shear = C_dv[i] * C_dv[j].transpose();
				K_shear.diagonal() += (dC_dvdv * C_shear) * Vector3d::Ones();
				K_shear *= -k_shear_f;

				// compute damping
				Matrix3d KD = -k_damping * dC_dvdv * Cdot * Matrix3d::Identity();
				K_shear += KD;

				// damping elements
				Matrix3d D_shear = -k_damping * C_dv[i] * C_dv[j].transpose();
				K_shear9.block(3 * i, 3 * j, 3, 3) = K_shear;		// we only fill the lower triangular matrix here
				D_shear9.block(3 * i, 3 * j, 3, 3) = D_shear;
			}
		}

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

						int row = idi + p * n; // corresponds to vertex id (i/j) and shifted by number of vertices for y and z coordinates (p)
						int col = idj + q * n;

						triK.push_back(Tri(row, col, K_shear9(3 * i + idp, 3 * j + idq)));			// ... these are always in the lower triangle
						triD.push_back(Tri(row, col, D_shear9(3 * i + idp, 3 * j + idq)));
						if (i != j) {
							row = idj + p * n; 														// only i and j are switched
							col = idi + q * n;
							triK.push_back(Tri(row, col, K_shear9(3 * i + idq, 3 * j + idp)));		// K_ij = K_ji.transpose() and symmetric!
							triD.push_back(Tri(row, col, D_shear9(3 * i + idq, 3 * j + idp)));		// but D_ij is not symmetric, why we need to do it this whole complicated way
						}
					}
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