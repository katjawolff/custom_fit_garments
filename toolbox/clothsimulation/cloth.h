#ifndef GARMENTSHAPE_CLOTH_H
#define GARMENTSHAPE_CLOTH_H

#include <Eigen/Sparse>
#include <igl/AABB.h>

#include "stretch_shear_forces.h"
#include "bend_quadratic_forces.h"
#include "constraint_forces.h"
#include "seam_forces.h"
#include "rest_shape.h"
#include "../linear_solver.h"
#include "../garmentcreation/garment_boundaries.h"

class Cloth {
private:
    const Eigen::Vector3d gravity = Eigen::Vector3d(0.0f, -9.81f, 0.0f);		// in m/s^2
    bool use_gravity = true;

	int n, m;					// number of vertices and faces
	double total_mass, mass;	// mass per vertex in g
	double k_stretch, k_shear, k_bend;

	Eigen::MatrixXi T;						// triangles = faces of the garment mesh
	Eigen::SparseMatrix<double> M, M_inverse, K, D;	// mass, stiffness, damping matrix	size = 3*n x 3*n 
	Eigen::MatrixXd X;						// current vertex positions					size = n x 3
	Eigen::VectorXd V, V_new;				// current and updated velocity of verts	size = 3*n
	Eigen::VectorXd F;						// current forces

    StretchShear stretchShear;
	Bend bend;
	Constraints constr;
	Seams seams;
    EvolveRestShape restshape;

    
#ifdef HAVE_CHOLMOD
    CholmodSolver solver;
#else
	LDLTSolver solver;						// NO: This solver needs a symmetric matrix. By default we only need to provide the lower part.
#endif
    
	// vertice constraints
	bool constraints = false;
	Eigen::VectorXd C;						// constraint vector
	GarmentBoundaries* boundary;
    std::vector<int> fixed_vert_id;
    std::vector<Eigen::Vector3d> fixed_vert_target;

	// collision detection
	bool collision = false;
	double EPS = 0.002f;							// in m (= 2 mm)
	igl::AABB<Eigen::MatrixXd, 3> tree_col;			// collision AABB tree for the mannequin
	Eigen::MatrixXd V_col, FN_col, VN_col, EN_col;	// vertices of the collision mesh
	Eigen::MatrixXi F_col, E_col;					// triangles = faces of the garment mesh / faces of the collision mesh
	Eigen::VectorXi EMAP_col;
	double collision_damping = 0.5;
	double friction = 0.5;

	void setCollisionMesh(Eigen::MatrixXd& V_col, Eigen::MatrixXi& F_col);
	
	void ComputeForces(float h);
	void ImplicitEuler(float h);
	void Collision();

public:
	Cloth(
		Eigen::MatrixXd& Vgarment,
		Eigen::MatrixXi& Fgarment,
		Eigen::MatrixXd& Vmannequin,
		Eigen::MatrixXi& Fmannequin,
		double total_mass,
		double K_stretch,
		double K_stretch_damping,
		double K_shear,
		double K_shear_damping,
		double K_bend,
		double K_bend_damping,
		double dist_to_body,
		bool with_collision);

	Eigen::MatrixXd getMesh();
    void setOffset(double dist_to_body);
    void setSimMesh(Eigen::MatrixXd& Vsim);
    void setGravity(bool use_gravity);

	void StepPhysics(double dt);
	void StepPhysics(double dt, Eigen::MatrixXd& V_col, Eigen::MatrixXi& F_col);
	void updateRestShape(Eigen::MatrixXd& V, double t_stretch, double t_compress, std::vector<double>& edge_adjustment);
	void setConstrainedVertices(GarmentBoundaries* boundary);
    void setSeamVertices(std::vector< std::pair<int,int> > seam_ids);

	Eigen::VectorXd ComputeStretchPerTriangle();
};

#endif //GARMENTSHAPE_CLOTH_H
