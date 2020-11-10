#ifndef GARMENTSHAPE_DIJKSTRA_H
#define GARMENTSHAPE_DIJKSTRA_H

#include <Eigen/Core>
#include <vector>
#include <set>

// Dijkstra's algorithm for shortest paths on a mesh, with multiple targets, using edge length
//
// Inputs:
//   V                #V by 3 list of vertex positions
//   VV               #V list of lists of incident vertices (adjacency list), e.g.
//                    as returned by igl::adjacency_list, will be generated if empty.
//   source           index of source vertex
//   targets          target vector set
//
// Output:
//   min_distance     #V by 1 list of the minimum distances from source to all vertices
//   previous         #V by 1 list of the previous visited vertices (for each vertex) - used for backtracking
//
int dijkstra(
    const Eigen::MatrixXd& V,
    const std::vector< std::vector<int> >& A,
    const int& source,
    const int& targets,
    Eigen::VectorXd& min_distance,
    Eigen::VectorXd& previous);

    // Backtracking after Dijkstra's algorithm, to find shortest path.
    //
    // Inputs:
    //   vertex           vertex to which we want the shortest path (from same source as above)
    //   previous         #V by 1 list of the previous visited vertices (for each vertex) - result of Dijkstra's algorithm
    //
    // Output:
    //   path             #P by 1 list of vertex indices in the shortest path from vertex to source
    //
    void dijkstra(const int& vertex, const Eigen::VectorXd& previous, std::vector<int>& path);

#endif
