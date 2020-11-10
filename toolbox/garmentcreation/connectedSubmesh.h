#ifndef GARMENTSHAPE_CONNECTEDSUBMESH_H
#define GARMENTSHAPE_CONNECTEDSUBMESH_H

#include <Eigen/Dense>

void getBiggestConnectedSubmesh(
	const Eigen::MatrixXd& V_in,
	const Eigen::MatrixXi& F_in,
	Eigen::MatrixXd& V_sub,
	Eigen::MatrixXi& F_sub
);

void getSubmesh(
	const Eigen::MatrixXd& V_in,
	const Eigen::MatrixXi& F_in,
	const Eigen::VectorXi& ids,
	int component,
	Eigen::MatrixXd& V_sub,
	Eigen::MatrixXi& F_sub
);

#endif