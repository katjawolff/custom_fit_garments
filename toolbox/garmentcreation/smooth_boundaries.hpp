#pragma once

#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "../linear_solver.h"

class SmoothBoundaries
{
    const Eigen::MatrixXi& F;
    LDLTSolver solver;
    std::vector< std::vector<int> > adj;
    
public:
    
    typedef std::vector< std::pair<int, Eigen::Vector3d> > ImplicitBoundary;
    
    SmoothBoundaries(const Eigen::MatrixXi& Fin);

    void initGeometry(const Eigen::MatrixXd& V);
    
    ImplicitBoundary smooth(const std::vector<int>& loop,
                            const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
    
    void getCoordinates(const Eigen::MatrixXd& V, const ImplicitBoundary& boudary, Eigen::MatrixXd& out);
};
