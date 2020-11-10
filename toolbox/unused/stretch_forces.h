#ifndef GARMENTSHAPE_STRETCH_FORCES_H
#define GARMENTSHAPE_STRETCH_FORCES_H

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

class Stretch{
private:
	double k_stretch;
	double k_damping;

	int n, m, n3, m81;							// number of vertices and faces
	Eigen::VectorXd a;							// triangle area of reference configuration
	std::vector< Eigen::MatrixXd > B;			// precomp
	std::vector< Eigen::Matrix2d > V_inverse;	// transformation matrix for each triangle of reference configuration
	Eigen::MatrixXi F;
	Eigen::VectorXd S;							// stretch per triangle - norm of 2d stretch vector

    bool tripletsInitialized = false;
    std::vector<Eigen::Triplet<double>> triK, triD;
    
	Eigen::SparseMatrix<double> K;				// out: stiffness matrix
	Eigen::SparseMatrix<double> D;				// out: damping matrix
	Eigen::VectorXi K_data, D_data;				// store the non-zero index data for K and D

public: 
	Stretch();

    const Eigen::SparseMatrix<double>& getK() const;
    const Eigen::SparseMatrix<double>& getD() const;
    
    
	void init(double k_stretch, double k_damping, Eigen::MatrixXd& V, Eigen::MatrixXi& F);
	void precompute_rest_shape(Eigen::MatrixXd& V);

	void compute_forces(
		const Eigen::MatrixXd& X,		// in: vertex positions
		const Eigen::VectorXd& V,		// in: velocitiy at vertex positions
        Eigen::VectorXd& F);	    	// out: force vector
	

	void compute_stretch(Eigen::VectorXd& S);

};

#endif //GARMENTSHAPE_STRETCH_SHEAR_FORCES_H
