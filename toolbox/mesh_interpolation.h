// See the papers:  Matrix Animation and Polar Decomposition by Shoemake and Duff
//					Deformation Transfer for Triangle Meshes by Sumner et al.
//					Thesis by Robert Sumner for the above paper

#ifndef GARMENTSHAPE_MESHINTERPOLATION_H
#define GARMENTSHAPE_MESHINTERPOLATION_H

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include "linear_solver.h"

class MeshInterpolator {
public:
	MeshInterpolator(
		const Eigen::MatrixXd& V_from,
		const Eigen::MatrixXd& V_to,
		const Eigen::MatrixXi& F);

	void interpolatedMesh(double p, Eigen::MatrixXd& V_interpolated);

	void precomputeInterpolatedMeshes(int steps);
	void getPrecomputedMesh(int step, Eigen::MatrixXd& V_p);

private:
	double positional_weight = 20.0;

	Eigen::MatrixXd V_from, V_to;
	Eigen::MatrixXi F;

	// precomputed meshes
	std::vector< Eigen::MatrixXd > V_precomp;
	int total_interpolations;

	// triangle rotation and stretch interpolation Q = R*S
	std::vector< Eigen::Matrix3d > U;	 // S = U*K*U2.transpose()
	std::vector< Eigen::Matrix3d > U2;
	std::vector< Eigen::Vector3d > K;
	std::vector< Eigen::Quaterniond > q; // quaternion of R

    LDLTSolver solver;
    
    Eigen::SparseMatrix<double> A;
	Eigen::VectorXi A_data;			 // store the non-zero index data for A
	Eigen::SparseMatrix<double> ATA; // we need to store this here, since cg only keeps a reference to this matrix

	int fixed_id;				// id of the one vertex that gets fixed in space
	Eigen::VectorXd A_remcol;	// removed column of A for that fixed vertex

	void precomputeA(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		int fixed_id,
		Eigen::SparseMatrix<double>& A,
		Eigen::VectorXd& A_remcol);

};

#endif //GARMENTSHAPE_CLOTH_H
