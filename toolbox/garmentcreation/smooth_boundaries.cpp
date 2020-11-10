#include "smooth_boundaries.hpp"

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/adjacency_list.h>

#include <fstream>
#include <queue>

SmoothBoundaries::SmoothBoundaries(const Eigen::MatrixXi& Fin)
: F(Fin)
{
    igl::adjacency_list(F, adj);
}

void SmoothBoundaries::initGeometry(const Eigen::MatrixXd& V)
{
    Eigen::SparseMatrix<double> M, L;
    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);

    const double t = 20 * std::pow(igl::avg_edge_length(V, F), 2);

    Eigen::SparseMatrix<double> A = M - t * L;
    
    solver.setSystem(A);
}

void SmoothBoundaries::getCoordinates(const Eigen::MatrixXd& V, const ImplicitBoundary& boundary, Eigen::MatrixXd& out)
{
    out.resize(boundary.size(), 3);
    
    for(int i = 0; i < boundary.size(); ++i)
    {
        const auto& bary = boundary[i];
        out.row(i) =  bary.second[0] * V.row(F(bary.first, 0)) + bary.second[1] * V.row(F(bary.first, 1)) + bary.second[2] * V.row(F(bary.first, 2));
    }
}

SmoothBoundaries::ImplicitBoundary
SmoothBoundaries::smooth(const std::vector<int>& loop,
                            const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    // build indicator function.
    // set all vertices on one side of the loop to 1, the rest is set -1 or 0 if it is part of the loop.
    
    Eigen::VectorXd b(V.rows());
    b.setConstant(-1.);
    
    for(int i : loop) b(i) = 0.;
    
    std::queue<int> qu;
   
    for(int i = 0; i < V.rows(); ++i)
    {
        if(b(i) != 0)
        {
            qu.push(i);
            b(i) = 1;
            break;
        }
    }
    
    while(!qu.empty())
    {
        int i = qu.front();
        qu.pop();
         
        for(int j : adj[i])
        {
            if(b(j) == -1.)
            {
                b(j) = 1;
                qu.push(j);
            }
        }
    }
    
    // smooth function
    solver.solve(b, b);
    
    // extract zero set
    
    ImplicitBoundary ret;
    
    for(int i = 0; i < F.rows(); ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            const double w0 = b(F(i, j));
            const double w1 = b(F(i, (j+1)%3));
            
            if(w0 < 0 ^ w1 < 0)
            {
                Eigen::Vector3d bary;
                bary(j) = w1 / (w1 - w0);
                bary((j+1)%3) = 1. - bary(j);
                bary((j+2)%3) = .0;

                ret.push_back(std::make_pair(i, bary));
            }
        }
    }
    
    return ret;
}




