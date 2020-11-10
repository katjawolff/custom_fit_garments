#include "cloth.h"

#include "../adjacency.h"
#include "../timer.h"

#include <iostream>
#include <chrono>
#include <memory>
#include <unsupported/Eigen/SparseExtra>
#include <igl/signed_distance.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

using namespace Eigen;
using namespace std;

typedef Triplet<double> Tri;

// ###############################
//   initialize cloth simulation 
// ###############################
Cloth::Cloth(
	MatrixXd & Vgarment,
	MatrixXi & Fgarment,
	MatrixXd & Vmannequin,
	MatrixXi & Fmannequin,
	double total_mass,
	double K_stretch,
	double K_stretch_damping,
	double K_shear,
	double K_shear_damping,
	double K_bend,
	double K_bend_damping,
	double dist_to_body,
	bool with_collision)
{
	EPS = dist_to_body;
	n = Vgarment.rows();
	int n3 = 3 * n;
	m = Fgarment.rows();

    F.resize(n3);

    // init vertex positions x and triangles
	X.resizeLike(Vgarment);
	X = Vgarment;
	T.resizeLike(Fgarment);
	T = Fgarment;

	this->mass = total_mass / n;
	this->total_mass = total_mass;

	//stretch.init(K_stretch, K_stretch_damping, X, T);
	//shear.init(K_shear, K_shear_damping, X, T);	
    stretchShear.init(K_stretch, K_stretch_damping, K_shear, K_shear_damping, X, T);    
	bend.init(K_bend, K_bend_damping, X, T);

	double k_constraints = 1e8;			// TODO
	double k_constraints_damping = 1.;	// TODO
	constr = *new Constraints();
	constr.init(k_constraints, k_constraints_damping, Vgarment.rows());
    seams.init(k_constraints, k_constraints_damping, Vgarment.rows());

    // init velocities
	V = VectorXd::Zero(n3);
	V_new = VectorXd::Zero(n3);

   // init sparse mass and stiffness matrix
	M.resize(n3, n3); 
	M.reserve(n3);
	for (int i = 0; i < n3; i++)
		M.insert(i, i) = mass;
	M_inverse.resize(n3, n3);
	M_inverse.reserve(n3);
	double mass_inverse = 1. / mass;
	for (int i = 0; i < n3; i++)
		M_inverse.insert(i, i) = mass_inverse;

    K.resize(n3,n3);
	D.resize(n3, n3);

	// collision with body mesh
	if (with_collision)
		setCollisionMesh(Vmannequin, Fmannequin);

	// rest shape
	restshape.init(X, T);
}

void Cloth::setOffset(double dist_to_body){
    EPS = dist_to_body;
}
void Cloth::setGravity(bool use_gravity){
    this->use_gravity = use_gravity;
}
void Cloth::setSimMesh(Eigen::MatrixXd& Vsim){
    X.resizeLike(Vsim);
    X = Vsim;
    V = VectorXd::Zero(Vsim.rows()*3);
    V_new = VectorXd::Zero(Vsim.rows()*3);
}

Eigen::MatrixXd Cloth::getMesh() {
	return X;
}

void Cloth::setCollisionMesh(MatrixXd& V_col, MatrixXi& F_col) {
	// collision with body mesh
	collision = true;

	this->V_col = V_col;
	this->F_col = F_col;

	tree_col.init(V_col, F_col);
	igl::per_face_normals(V_col, F_col, FN_col);
	igl::per_vertex_normals(V_col, F_col, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN_col, VN_col);
	igl::per_edge_normals(V_col, F_col, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN_col, EN_col, E_col, EMAP_col);
}

void Cloth::updateRestShape(MatrixXd& V, double t_stretch, double t_compress, vector<double>& edge_adjustment) {
	
    std::cout << "Cloth::updateRestShape" << std::endl;
    Timer t("updateRestShape");
    
    restshape.adjust_restshape(X, V_col, t_stretch, t_compress, edge_adjustment, V);
    
    t.printTime("adjust_restshape");
    
	// precomputation for different forces
    stretchShear.precompute_rest_shape(V);
	bend.precompute_rest_shape(V);

     t.printTime("precompute_rest_shape");
    


    std::cout << "Cloth::updateRestShape done" << std::endl;
}

// ###############################
//   one simulation step 
// ###############################
void Cloth::StepPhysics(double dt) {
	ComputeForces(dt);
	ImplicitEuler(dt);
    Timer t;
	Collision();
    t.printTime("collision");
}

void Cloth::StepPhysics(double dt, MatrixXd & V_col, MatrixXi & F_col) {
    // The avatar moved, so we have to adjust the position of fixed boundary vertices
    MatrixXd Vb;
    vector<int> constrained_verts;
    boundary->getGarmentBoundaryFixedVertexPositions(V_col, Vb, constrained_verts);

    vector<Vector3d> constrained_verts_target(Vb.rows() + fixed_vert_target.size());
    for (int i = 0; i < Vb.rows(); i++) constrained_verts_target[i] = Vb.row(i);
    for (int i = 0; i < fixed_vert_id.size(); i++) constrained_verts_target[Vb.rows() + i] = fixed_vert_target[i];

    vector<int> constrained_verts_all(constrained_verts.size() + fixed_vert_id.size());
    for (int i = 0; i < constrained_verts.size(); i++) constrained_verts_all[i] = constrained_verts[i];
    for (int i = 0; i < fixed_vert_id.size(); i++) constrained_verts_all[constrained_verts.size() + i] = fixed_vert_id[i];

    constr.precompute_rest_shape(constrained_verts_all, constrained_verts_target);



	ComputeForces(dt);
	ImplicitEuler(dt);
	setCollisionMesh(V_col,F_col);
	Collision();
}

// ###############################
//   compute forces - F and K
// ###############################

VectorXd Cloth::ComputeStretchPerTriangle() {
	VectorXd S = stretchShear.getStretch();
    
	double total_stretch = (S - VectorXd::Ones(S.rows())).array().abs().matrix().sum();
	cout << "Integrated stretch: " << total_stretch << endl;
    return S;
}

void Cloth::ComputeForces(float h) {

	// --- gravity ---
	if(use_gravity) {
        F.block(0, 0, n, 1).setZero();
        F.block(n, 0, n, 1).setConstant(mass * gravity(1));
        F.block(2 * n, 0, n, 1).setZero();
    } else {
	    F = VectorXd::Zero(n*3);
	}
    
    Timer t("forces");
    
    // --- STRETCH SHEAR ---
    stretchShear.compute_forces(X, V, F);
    t.printTime("compute stetch shear");
    
	// --- BEND ---
	bend.compute_forces(X, V, F);
    t.printTime("compute bend");
    
	// --- CONSTRAINTS ---
	SparseMatrix<double> K_constr, D_constr;
	constr.compute_forces(X, V, F, K_constr, D_constr);
    t.printTime("compute constr");

    // --- SEAMS ---
    SparseMatrix<double> K_seams, D_seams;
    seams.compute_forces(X, V, F, K_seams, D_seams);
    t.printTime("compute seam forces");

    // everything added
    K = stretchShear.getK() + bend.getK() + K_constr + K_seams;
    D = stretchShear.getD() + bend.getD() + D_constr + D_seams;
    
    t.printTime("assemble");
}

// ###############################
//   compute constraints
// ###############################

void Cloth::setConstrainedVertices(GarmentBoundaries* boundary) {
	constraints = true;
	this->boundary = boundary;

	MatrixXd Vb;
	vector<int> constrained_verts;
	boundary->getGarmentBoundaryFixedVertexPositions(V_col, Vb, constrained_verts);

	/*vector<bool> fixed_boundary_vert(X.rows(), false);
	for (auto i : constrained_verts) {
		fixed_boundary_vert[i] = true;
		cout << "fixed vertex" << endl;
	}*/

    //restshape.setFixedVertices(fixed_boundary_vert);


    // set position of fixed boundary vertices
    //MatrixXd Vb;
    //vector<int> constrained_verts;
    //boundary->getGarmentBoundaryFixedVertexPositions(V_col, Vb, constrained_verts);

    //t.printTime("getGarmentBoundaryFixedVertexPositions");

    vector<Vector3d> constrained_verts_target(Vb.rows() + fixed_vert_target.size());
    for (int i = 0; i < Vb.rows(); i++) constrained_verts_target[i] = Vb.row(i);
    for (int i = 0; i < fixed_vert_id.size(); i++) constrained_verts_target[Vb.rows() + i] = fixed_vert_target[i];

    vector<int> constrained_verts_all(constrained_verts.size() + fixed_vert_id.size());
    for (int i = 0; i < constrained_verts.size(); i++) constrained_verts_all[i] = constrained_verts[i];
    for (int i = 0; i < fixed_vert_id.size(); i++) constrained_verts_all[constrained_verts.size() + i] = fixed_vert_id[i];

    constr.precompute_rest_shape(constrained_verts_all, constrained_verts_target);
    //t.printTime("precompute_rest_shape constr");

}

void Cloth::setSeamVertices(vector< pair<int,int> > seam_ids){
    seams.precompute_rest_shape(seam_ids);
}

// ###############################
//   solver: backward/implicit euler
// ###############################
void Cloth::ImplicitEuler(float h) {

    Timer t;
    
	SparseMatrix<double> A = M - h * (D + h * K);				   // only lower half is filled
	VectorXd b = M * V + h * (F - D.selfadjointView<Lower>() * V); // not b from the paper, but behaves better and we solve directly for V
    t.printTime("final assemble");
    int cnt = 0;
      
	while (!solver.setSystem(A.selfadjointView<Lower>())) {
        ++cnt;
        h /= 2.0;
    
		cout << "  A not PSD. Reduce time step: " << h  << endl;
		if (h < 1.e-5) {
			cout << "  h is too small. Stop the simulation." << endl;
			return;
		}
		A = M - h * (D + h * K);
		b = M * V + h * (F - D.selfadjointView<Lower>() * V);
    }

    t.printTime("factorize (" + std::to_string(1 + cnt) + " times)");

    
	//solver.setSystem(A);
    solver.solve(b, V_new);
    t.printTime("solve");
    
    //V_new += V;				// for the b of the paper we get dV, and need to: V + dV
/*
	cout << A.row(0) << endl;
	cout << "Sum of row: " << A.row(0).sum() << endl;
	cout << "b(0) " << b(0) << endl;
	cout << "Vnew(0) " << V_new(0) << endl;
	double diff = (A * V_new - b).norm();
	cout << "error " << diff << endl;
	//Eigen::saveMarket
	string folder_path = igl::file_dialog_save();
	Eigen::saveMarket(A, folder_path);
	folder_path = igl::file_dialog_save();
	Eigen::saveMarketVector(b, folder_path);
*/
    
	X.col(0) += h * V_new.head(n);
	X.col(1) += h * V_new.segment(n,n);
	X.col(2) += h * V_new.tail(n);

	V = V_new;
    
    t.printTime("update position + velocity");
}

// ###############################
//   collision handling
// ###############################
void Cloth::Collision() {

	if (collision) {
		// get distance of each garment point to the body mesh
		VectorXd S;
		VectorXi I;
		MatrixXd C, N;
		igl::signed_distance_pseudonormal(X, V_col, F_col, tree_col, FN_col, VN_col, EN_col, EMAP_col, S, I, C, N);

		// now check each point if it penetrates the surface
		// S has the distances of the vertex to the surface
		// N has the normal direction so we can push away in this direction
		// C contains the clostest point on our mannequin
		MatrixXd Xnew(X.rows(),3); Xnew = X;
		for (int v = 0; v < n; v++) {
			if (S(v) < EPS) {
				Vector3d normal = N.row(v).normalized();	// unfortunately N seems not to be correctly normalized sometimes
				Xnew.row(v) += (-S(v) + EPS) * normal;		// adjust position

				Vector3d vel;
				for (int i = 0; i < 3; i++)					
					vel(i) = V(v + i * n);			

				// adjust velocity
				Vector3d n_vel = normal.dot(vel) * normal;	// get velocity along collision normal
				Vector3d t_vel = vel - n_vel;				// translational velocity
				vel = t_vel - collision_damping * n_vel;	// reflect velocity along collision normal
				for (int i = 0; i < 3; i++)
					V(v + i * n) = vel(i);
			}

			// check with ground plane
			//if (X(v, 1) < 0) {
			//	X(v, 1) = 0;
			//}
		}

		// double check if vertices are really outside the mesh - fix for avatar self intersections
		igl::SignedDistanceType type = igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER;
		igl::signed_distance(Xnew,V_col,F_col,type,S,I,C,N);
		fixed_vert_id.clear();
		fixed_vert_target.clear();
        for (int v = 0; v < n; v++) {
            if (S(v) < 0) {
                fixed_vert_id.push_back(v);
                fixed_vert_target.push_back(X.row(v));
            }
        }


        X = Xnew;
	}
}

