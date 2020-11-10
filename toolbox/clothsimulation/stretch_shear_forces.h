#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class StretchShear{
private:
    double k_stretch;
    double k_stretch_damping;

    double k_shear;
    double k_shear_damping;

    int n, m, n3, m81;                            // number of vertices and faces
    Eigen::VectorXd a;                            // triangle area of reference configuration
    std::vector< Eigen::MatrixXd > B;            // precomp
    std::vector< Eigen::Matrix2d > V_inverse;    // transformation matrix for each triangle of reference configuration
    Eigen::MatrixXi F;
    Eigen::VectorXd S;                            // stretch per triangle - norm of 2d stretch vector

    bool tripletsInitialized = false;
    Eigen::VectorXd triK, triD, triF;
    
    Eigen::SparseMatrix<double> K;                // out: stiffness matrix
    Eigen::SparseMatrix<double> D;                // out: damping matrix
    Eigen::VectorXi K_data;                       // store the non-zero index data for K and D

public:
    StretchShear();

    const Eigen::SparseMatrix<double>& getK() const;
    const Eigen::SparseMatrix<double>& getD() const;
    const Eigen::VectorXd& getStretch() const;
      
    void init(double k_stretch, double k_stretch_damping, double k_shear, double k_shear_damping, Eigen::MatrixXd& V, Eigen::MatrixXi& F);
    void precompute_rest_shape(Eigen::MatrixXd& V);

    void compute_forces(
        const Eigen::MatrixXd& X,        // in: vertex positions
        const Eigen::VectorXd& V,        // in: velocitiy at vertex positions
        Eigen::VectorXd& F);             // out: force vector
};
