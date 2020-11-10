#ifndef GARMENTSHAPE_SHEAR_FORCES_H
#define GARMENTSHAPE_SHEAR_FORCES_H

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

class Shear{
private:
	double k_shear;
	double k_damping;

	int n, m, n3, m81;							// number of vertices and faces
	Eigen::VectorXd a;							// triangle area of reference configuration
	std::vector< Eigen::MatrixXd > B;			// precomp
	std::vector< Eigen::Matrix2d > V_inverse;	// transformation matrix for each triangle of reference configuration
	Eigen::MatrixXi F;

	Eigen::SparseMatrix<double> K;				// out: stiffness matrix
	Eigen::SparseMatrix<double> D;				// out: damping matrix
	Eigen::VectorXi K_data, D_data;				// store the non-zero index data for K and D

public: 
	Shear();

	void init(double k_shear, double k_damping, Eigen::MatrixXd& V, Eigen::MatrixXi& F);
	void precompute_rest_shape(Eigen::MatrixXd& V);

	void compute_forces(
		const Eigen::MatrixXd& X,		// in: vertex positions
		const Eigen::VectorXd& V,		// in: velocitiy at vertex positions
		Eigen::VectorXd& F,				// out: force vector
		Eigen::SparseMatrix<double>& K,	// out: stiffness matrix 
		Eigen::SparseMatrix<double>& D	// out: damping matrix
	);

};

#endif //GARMENTSHAPE_SHEAR_SHEAR_FORCES_H
