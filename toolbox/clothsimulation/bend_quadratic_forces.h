#ifndef GARMENTSHAPE_BEND_FORCES_H
#define GARMENTSHAPE_BEND_FORCES_H

#include <vector>
#include <Eigen/Sparse>

class Bend {
private:
	double k_bend;
	double k_damping;

	int n;								// number of vertices
	Eigen::MatrixXi E4;					// four vertices around each interiour edge
    std::vector< Eigen::Vector4d > F_local;

    bool tripletsInitialized = false;
    Eigen::VectorXi K_data;
    std::vector<Eigen::Triplet<double>> triK, triD;

    std::vector< Eigen::Matrix4d > Q;	// precomputed matrix Q for each interiour edge
	Eigen::SparseMatrix<double> K, D;	// precomputed constant stiffness and damping matrix

	double cotTheta(const Eigen::Vector3d v, const Eigen::Vector3d w);
	void ComputeLocalStiffness(const std::vector< Eigen::Vector3d>& x, Eigen::Matrix4d& Q);
public:
	Bend();

	void init(
		const double k_bend,
		const double k_damping,
		const Eigen::MatrixXd& X,	// in: vertex positions
		const Eigen::MatrixXi& T	// in: mesh triangles
	);

    const Eigen::SparseMatrix<double>& getK() const;
    
    const Eigen::SparseMatrix<double>& getD() const;
    
	void precompute_rest_shape(const Eigen::MatrixXd& X);

	void compute_forces(
		const Eigen::MatrixXd& X,		// in: vertex positions
		const Eigen::VectorXd& V,		// in: velocities
        Eigen::VectorXd& F);	   	    // out: forces

	
};

#endif //GARMENTSHAPE_BEND_FORCES_H
