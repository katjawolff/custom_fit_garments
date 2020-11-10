#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../linear_solver.h"
#include "../garmentcreation/garment_boundaries.h"

#include <igl/AtA_cached.h>

class EvolveRestShape {
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


    // arap data
    Eigen::MatrixXd cotanEntries;
    
    
    
    #ifdef HAVE_CHOLMOD
        CholmodSolver solver2;
    #else
        LDLTSolver solver2;
    #endif
    
    
    #ifdef HAVE_CHOLMOD
        CholmodSolver solver;
    #else
        LDLTSolver solver;
    #endif
    
    
    Eigen::SparseMatrix<double> J0;
    Eigen::SparseMatrix<double> JTJ0;
    igl::AtA_cached_data JTJcache;
    Eigen::SparseMatrix<double> JTJ;
    
    Eigen::SparseMatrix<double> L;
    Eigen::MatrixXd Vinit;
    
     // edge lenght weight is always 1.
    const double eps = 1e-7;
    
    double alpha =  1.e-2;       // how to weight angle constraints
    double lambda = 1.e-6;      // how to weight positional constraint

    

public:
    EvolveRestShape();

    void init(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
    
    void jacobianConstant(
        const Eigen::MatrixXd& Xi,
        const std::vector< int >& edge_constraints
    );
    
    void jacobianUpdate(
        const Eigen::MatrixXd& Xi,
        const std::vector< int >& edge_constraints,
        Eigen::SparseMatrix<double>& J
    );
        
    void energy(
        const Eigen::MatrixXd& Xi,
        const Eigen::MatrixXd& X0,
        const std::vector<double>& length_target,
        const std::vector<int>& edge_constraints,
        Eigen::VectorXd& f
    );
    
    double checkFiniteDifferences(
        const Eigen::MatrixXd& Xi,
        const Eigen::MatrixXd& X0,
        const std::vector< double >& length_target,
        const std::vector< int >& edge_constraints
    );
    
    void setFixedVertices(const std::vector<bool>& fixed_boundary_vert);

    void arap(std::vector<Eigen::Matrix2d>& X0, Eigen::MatrixXd& Xi, int iters);
    
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

