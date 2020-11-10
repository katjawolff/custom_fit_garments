#ifndef GARMENTSHAPE_SHAPE_H
#define GARMENTSHAPE_SHAPE_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../linear_solver.h"
#include "../garmentcreation/garment_boundaries.h"

class RestShape {
private:

    // adjacency information
    Eigen::MatrixXi F;
    Eigen::MatrixXi E;						// Edges - for each both vertex IDs - gives a global edge orientation
    Eigen::MatrixXi E4;						// four vertices around each interiour edge, see adjacency.h. First three always exist. v3 might be -1 for border edges.
    Eigen::MatrixXi EF6;					// for each edge: for the first and second face: for all three vertex ids from the edge-with-4-verts map into ids of the face {0,1,2}
    Eigen::MatrixXi EF_adj;					// for each edge: the two adjacent faces. First always exists. -1 if there is no second face.
    std::vector< std::vector<int> > FE_adj;	// for each face: list of adjacent edges, maps into E4, EF6, EF_adj and E, edges are sorted as opposites to v0, v1, v2
    std::vector<bool> fixed_boundary_vert;  // true for each vertex that is fixed due to a fixed boundary
    std::vector<bool> fixed_boundary_edge;  // true for each edge ...
    int verts;
    int faces;
    int edges;
    int interior_edges;

    void shape_energy(
        const Eigen::MatrixXd& X_current,
        const Eigen::MatrixXd& X_restshape,
        const std::vector< double >& length_target,
        const std::vector< double >& angle_target,
        int edge_constraints_count,
        const std::vector< int >& edge_constraints,
        Eigen::VectorXd& f,
        Eigen::SparseMatrix<double>& J,
        bool with_jacobian
    );

public:
    RestShape();

    void init(const Eigen::MatrixXi& F);
    
    void setFixedVertices(const std::vector<bool>& fixed_boundary_vert);

    void adjust_restshape(
        const Eigen::MatrixXd& X_simulation,
        const Eigen::MatrixXd& Vm,
        const double t_stretch,
        const double t_compress,
        Eigen::MatrixXd& X_restshape
    );
    void adjust_restshape(
        const Eigen::MatrixXd& X_simulation,
        const Eigen::MatrixXd& Vm,
        const double t_stretch,
        const double t_compress,
        const std::vector<double>& edge_adjustment,
        Eigen::MatrixXd& X_restshape
    );
};

#endif