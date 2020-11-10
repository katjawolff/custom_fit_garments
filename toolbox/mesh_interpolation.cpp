#include "mesh_interpolation.h"

#include <igl/polar_dec.h>
#include <igl/svd3x3.h>
#include <igl/AABB.h>
#include <igl/signed_distance.h>
#include <igl/centroid.h>
#include <igl/sparse_cached.h>

using namespace std;
using namespace Eigen;

typedef Triplet<double> Tri;

Vector3d face_centroid(const Vector3i& f, const MatrixXd& V) {
	return (V.row(f(0)) + V.row(f(1)) + V.row(f(2))) / 3.0;
}

MeshInterpolator::MeshInterpolator(
	const MatrixXd& V_from,		// mesh vertices of the source mesh
	const MatrixXd& V_to,		// mesh vertices of the target mesh
	const MatrixXi& F			// faces of both meshes
) {
	assert(V_from.rows() == V_to.rows());

	this->V_from = V_from;
	this->V_to = V_to;
	this->F = F;

	U.resize(F.rows());
	U2.resize(F.rows());
	K.resize(F.rows());
	q.resize(F.rows());

	// find a fixed vertex 
	// use the vertex that has the smallest distance to its final destination
	double smallest_dist = 1000000;
	fixed_id = -1;
	for (int v = 0; v < V_from.rows(); v++) {
		double dist = (V_from.row(v) - V_to.row(v)).norm();
		if (dist < smallest_dist) {
			smallest_dist = dist;
			fixed_id = v;
		}
	}

	// precomputation
	for (int f = 0; f < F.rows(); f++) {
		Vector3d v1 = V_from.row(F(f, 0));
		Vector3d v2 = V_from.row(F(f, 1));
		Vector3d v3 = V_from.row(F(f, 2));

		Vector3d v1_def = V_to.row(F(f, 0));
		Vector3d v2_def = V_to.row(F(f, 1));
		Vector3d v3_def = V_to.row(F(f, 2));

		// calculate the 4th vertex for this face
		Vector3d a = (v2 - v1).cross(v3 - v1);
		Vector3d v4 = v1 + a / sqrt(a.norm());
		Vector3d a_def = (v2_def - v1_def).cross(v3_def - v1_def);
		Vector3d v4_def = v1_def + a_def / sqrt(a_def.norm());

		// calculate matrices V, V_def and Q
		Matrix3d V, V_def, Q;
		V.col(0) = v2 - v1;
		V.col(1) = v3 - v1;
		V.col(2) = v4 - v1;
		V_def.col(0) = v2_def - v1_def;
		V_def.col(1) = v3_def - v1_def;
		V_def.col(2) = v4_def - v1_def;
		Q = V_def * V.inverse();

		// polar decomposition
		Matrix3d R, S;
		igl::polar_dec(Q, R, S);

		// decompose S
		Matrix3d U, U2;
		Vector3d K;
		igl::svd3x3(S, U, K, U2);
		this->K[f] = K;
		this->U[f] = U;
		this->U2[f] = U2;

		// decompose R
		Quaterniond q(R);
		this->q[f] = q;
	}

	// calculate A
	precomputeA(V_from, F, fixed_id, A, A_remcol);
	ATA = A.transpose() * A;	
    solver.setSystem(ATA);
}

void MeshInterpolator::precomputeA(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	int fixed_id,
	Eigen::SparseMatrix<double>& A,
	Eigen::VectorXd& A_remcol
){
	// reserve space
	int n = V.rows();
	int m = F.rows();
	vector< Tri > A_entries;

	//A.resize(3 * m, n - 1);
	//A.reserve(9 * m);
	A_entries.reserve(9 * m);
	A_remcol = VectorXd::Zero(3 * m);

	for (int f = 0; f < m; f++) {
		MatrixXd W(3, 2);
		W.col(0) = V.row(F(f, 1)) - V.row(F(f, 0));
		W.col(1) = V.row(F(f, 2)) - V.row(F(f, 0));

		HouseholderQR<MatrixXd> qr(W);
		Matrix3d Q_qr = qr.householderQ();
		MatrixXd R_qr = Q_qr.transpose() * W;
		MatrixXd T = R_qr.block<2, 2>(0, 0).inverse() * Q_qr.block<3, 2>(0, 0).transpose();

		// fill A - refer to page 65 of Robert Sumners Thesis
		// check each time if we are in the removed column instead
		int i1 = F(f, 0);
		int i2 = F(f, 1);
		int i3 = F(f, 2);

		if (i2 != fixed_id) {
			if (i2 > fixed_id) i2 -= 1;
			for (int i = 0; i < 3; i++)
				A_entries.push_back(Tri(3 * f + i, i2, T(0, i)));
		}
		else
			for (int i = 0; i < 3; i++)
				A_remcol(3 * f + i) = T(0, i);
		if (i3 != fixed_id) {
			if (i3 > fixed_id) i3 -= 1;
			for (int i = 0; i < 3; i++)
				A_entries.push_back(Tri(3 * f + i, i3, T(1, i)));
		}
		else
			for (int i = 0; i < 3; i++)
				A_remcol(3 * f + i) = T(1, i);
		if (i1 != fixed_id) {
			if (i1 > fixed_id) i1 -= 1;
			for (int i = 0; i < 3; i++)
				A_entries.push_back(Tri(3 * f + i, i1, -T(0, i) - T(1, i)));
		}
		else
			for (int i = 0; i < 3; i++)
				A_remcol(3 * f + i) = -T(0, i) - T(1, i);
	}

	// build the sparse matrices from the triplets
	if (A_data.rows() == 0) {
		A = SparseMatrix<double>(3 * m, n - 1);
		igl::sparse_cached_precompute(A_entries, A_data, A);
		cout << "initialize matrix structure" << endl;
	}
	else 
		igl::sparse_cached(A_entries, A_data, A);

	//A.setFromTriplets(A_entries.begin(), A_entries.end());
}

void MeshInterpolator::precomputeInterpolatedMeshes(int steps) {
	total_interpolations = steps;	// e.g. if steps = 3 : p = 0, 0.5, 1
	V_precomp.resize(steps);

	for (int i = 0; i < steps; i++) {
		double p = float(steps-1 -i) / float(steps-1);
		interpolatedMesh(p, V_precomp[i]);
	}
}

void MeshInterpolator::getPrecomputedMesh(int step, MatrixXd& V_p) {
	if (step >= 0 && step < total_interpolations)
		V_p = V_precomp[step];
}

void MeshInterpolator::interpolatedMesh(double p, MatrixXd & V_interpolated) {
	assert(p >= 0 && p <= 1);

	// if p is close to 1, return just the V_to mesh. Helps with numerical errors. 
	if (p >= 0.9999) {
		V_interpolated = V_to;
		return;
	}

	// ======================
	//  mesh interpolation

	V_interpolated = MatrixXd::Zero(V_from.rows(), 3);

	// build the right hand side of the system
	MatrixXd RHS_Q = MatrixXd::Zero(3 * F.rows(), 3);

	for (int f = 0; f < F.rows(); f++) {
		// interpolate S
		Vector3d K_inter = (1.0 - p) * Vector3d::Ones() + p * K[f];
		Matrix3d S_inter = U[f] * K_inter.asDiagonal() * U2[f].transpose();

		// interpolate R
		Quaterniond qI(Matrix3d::Identity());
		Quaterniond q_inter = qI.slerp(p, q[f]);
		MatrixXd R_inter = q_inter.toRotationMatrix();

		Matrix3d Q_inter = R_inter * S_inter;

		RHS_Q.block<3, 3>(3 * f, 0) = Q_inter.transpose();
	}

	// substract the missing vertex and row from the right hand side
	// refer to page 68 of Robert Sumners thesis
	MatrixXd RHS = RHS_Q;
	Vector3d v_fixed = (1.0 - p) * V_from.row(fixed_id) + p * V_to.row(fixed_id);
	RHS.col(0) -= v_fixed(0) * A_remcol;
	RHS.col(1) -= v_fixed(1) * A_remcol;
	RHS.col(2) -= v_fixed(2) * A_remcol;

	// Build the linear system:
	RHS = A.transpose() * RHS;

	MatrixXd V_solved;
	solver.solve(RHS, V_solved);

	// plug in the one fixed vertex into the solution
	V_interpolated.block(0, 0, fixed_id, 3) = V_solved.block(0, 0, fixed_id, 3);
	V_interpolated.row(fixed_id) = v_fixed;
	V_interpolated.block(fixed_id + 1, 0, V_from.rows() - fixed_id - 1, 3) = V_solved.block(fixed_id, 0, V_from.rows() - fixed_id - 1, 3);
}
