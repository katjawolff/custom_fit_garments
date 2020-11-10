#ifndef GARMENTSHAPE_CONSTRAINT_FORCES_H
#define GARMENTSHAPE_CONSTRAINT_FORCES_H

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

class Constraints {
private:
	double k_constraints;
	double k_damping;

	int n, m, n3, c;							// number of vertices and faces

	std::vector<int> constrained_vert_id;
	std::vector<Eigen::Vector3d> constrained_vert_target;

	Eigen::SparseMatrix<double> K;				// out: stiffness matrix
	Eigen::SparseMatrix<double> D;				// out: damping matrix
	Eigen::VectorXi K_data, D_data;				// store the non-zero index data for K and D
	bool k_redo = true;

public:
	Constraints();

	void init(double k_constraints, double k_damping, int n);
	void precompute_rest_shape(std::vector<int> constrained_vert_id, std::vector<Eigen::Vector3d> constrained_vert_target);

	void compute_forces(
		const Eigen::MatrixXd& X,		// in: vertex positions
		const Eigen::VectorXd& V,		// in: velocitiy at vertex positions
		Eigen::VectorXd& F,				// out: force vector
		Eigen::SparseMatrix<double>& K,	// out: stiffness matrix 
		Eigen::SparseMatrix<double>& D	// out: damping matrix
	);
};

#endif //GARMENTSHAPE_CONSTRAINT_FORCES_H
