#ifndef GARMENTSHAPE_BEND_FORCES_H
#define GARMENTSHAPE_BEND_FORCES_H

#include <vector>
#include <Eigen/Sparse>

class Bend {
private:
	double k_bend;
	double k_damping;
	double epsilon = 1E-15f;

	int n, n3, m, no_edges, k_entries;  // number of vertices, *3, faces, edges, entries of the K matrix
	Eigen::MatrixXi T;					// faces

	// adjacency information
	Eigen::MatrixXi E;						// Edges - for each both vertex IDs - gives a global edge orientation
	Eigen::MatrixXi E4;						// four vertices around each interiour edge, see adjacency.h. First three always exist. v3 might be -1 for border edges.
	Eigen::MatrixXi EF6;					// for each edge: for the first and second face: for all three vertex ids from the edge-with-4-verts map into ids of the face {0,1,2}
	Eigen::MatrixXi EF_adj;					// for each edge: the two adjacent faces. First always exists. -1 if there is no second face.
	std::vector< std::vector<int> > FE_adj;	// for each face: list of adjacent edges, maps into E4, EF6, EF_adj and E, edges are sorted as opposites to v0, v1, v2

	// precompputed values
	Eigen::VectorXd phi_bar, phi_hat, phi_d;// tangent of theta/2 of the reference configuration // of the last time step
	Eigen::VectorXd a;

	Eigen::SparseMatrix<double> K;				// out: stiffness matrix
	Eigen::SparseMatrix<double> D;				// out: damping matrix
	Eigen::VectorXi K_data, D_data;				// store the non-zero index data for K and D

public:
	Bend();

	void init(
		const double k_bend,
		const double k_damping,
		const Eigen::MatrixXd& X,	// in: vertex positions
		const Eigen::MatrixXi& T	// in: mesh triangles
	);

	void precompute_rest_shape(const Eigen::MatrixXd& X);

	void compute_forces(
		const Eigen::MatrixXd& X,		// in: vertex positions
		const double timestep,		// in: delta t
		Eigen::VectorXd& F,				// out: forces
		Eigen::SparseMatrix<double>& K,	// out: stiffness matrix
		Eigen::SparseMatrix<double>& D	// out: damping matrix
	);
};

#endif //GARMENTSHAPE_BEND_FORCES_H