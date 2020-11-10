#ifndef GARMENTSHAPE_BOUNDARIES_H
#define GARMENTSHAPE_BOUNDARIES_H

#include <vector>
#include <Eigen/Dense>
#include "smooth_boundaries.hpp"


class GarmentBoundaries {
private:
	// we do not store vertices of the mesh here, as these can change with different poses
	// when new points are added to the boundary, we instead always work with the current vertex positions
	Eigen::MatrixXi F;								// faces of the mesh the boundaries are defined on
	std::vector< std::vector<int> > A;				// adjacency list of the mesh
	//std::vector< std::vector<int> > boundary_verts;	// stores all closed boundaries on the mesh
    
    SmoothBoundaries smooth;
    
    std::vector<int> activeBoundary;
    std::vector< SmoothBoundaries::ImplicitBoundary > boundaries;

	// garment mesh boundaries to boundaries correspondence
	std::vector< int > mesh_boundary_ids;			// vertex ids of the boundary of the garment (!!) mesh that has been created from boundaries
	std::vector< int > boundary_ids;				// boundary id of the boundary where an edge is closest to above
	std::vector< int > boundary_v_ids;				// vertex id of the first vertex of the edge of a boundary closest to above
	std::vector< double > boundary_p;				// where on the boundary edge lies the mesh boundary vertex? p is between 0 and 1
    
	// fixed boundaries
	std::vector<int> fixed;							// maps into "boundaries" - fixed in the cloth simulation
    
	void cylinder(Eigen::MatrixXd& Vcyl, Eigen::MatrixXi& Fcyl); // create a cylinder mesh
	void garmentFromBoundaries(const Eigen::MatrixXd& V, int v_start, Eigen::MatrixXd& Vg, Eigen::MatrixXi& Fg, std::vector<bool>& visited, double refinement);

public:
	GarmentBoundaries(Eigen::MatrixXi & F);

	void addPointsToBoundary(const Eigen::MatrixXd& V, int v_id);
	void closeBoundary(const Eigen::MatrixXd& V);	// for circular boundaries
	void endBoundary(const Eigen::MatrixXd& V);		// for boundaries from garment edge to garment edge
	void garmentFromBoundaries(const Eigen::MatrixXd& V, int v_start, Eigen::MatrixXd& Vg, Eigen::MatrixXi& Fg, double refinement);
	void cutGarmentAlongBoundaries(const Eigen:: MatrixXd& Vg, std::vector<Eigen::MatrixXd>& Vg_list, std::vector<Eigen::MatrixXi>& Fg_list);
	void markClosestBoundaryAsFixed(const Eigen::MatrixXd& Vm, Eigen::Vector3d& v);
	void getGarmentBoundaryFixedVertexPositions(const Eigen::MatrixXd& Vm, Eigen::MatrixXd& Vb, std::vector<int>& boundary_ids);
	void getCylindersAroundBoundaryEdges(const Eigen::MatrixXd& V, double radius, Eigen::MatrixXd& Vb, Eigen::MatrixXi& Fb);
	void createCorrespondences(const Eigen::MatrixXd& V, Eigen::MatrixXd& Vg, Eigen::MatrixXi& Fg);
	
	void saveBoundaries(const std::string fname);
	void loadBoundaries(const std::string fname);
	void deleteLast();
	void deleteAll();
	int numberOfBoundaries();
};

#endif
