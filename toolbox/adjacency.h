#ifndef MESHMOULD_ADJACENCY_H
#define MESHMOULD_ADJACENCY_H

#include <vector>
#include <Eigen/Core>

// special methods for this project:
// for each non-boundary edge, we collect all 4 corresponding vertex ids in a specific order
//         x2
//         /\
//        /  \
//     e1/    \e3
//      /  t0  \
//     /        \
//    /    e0    \
//  x0------------x1
//    \          /
//     \   t1   /
//      \      /
//     e2\    /e4
//        \  /
//         \/
//         x3
//
// this one is for the bending energy
void createFacePairEdgeListWith4VerticeIDs(
	const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& E,
    const std::vector< std::vector<int> >& vf_adj,
    Eigen::MatrixXi& E4,
    Eigen::MatrixXi& EF6,
    Eigen::MatrixXi& ef_adj
);

// this one is slightly different for the quadratic bending energy
void createFacePairEdgeListWith4VerticeIDs(
    const Eigen::MatrixXi& F,
    Eigen::MatrixXi& E4
);

// lists for each vertex the adjacent faces' indices
void createVertexFaceAdjacencyList(
        const Eigen::MatrixXi &F,
        std::vector< std::vector<int> > &adjecencyList
);

void createFaceEdgeAdjecencyList(
        const Eigen::MatrixXi & F,
        const Eigen::MatrixXi & E,
        const std::vector< std::vector<int> > & vertexFaceAdjecencyList,
        std::vector< std::vector<int> > & faceEdgeAdjecencyList
);

void createVertexEdgeAdjecencyList(
        const Eigen::MatrixXi & E,
        std::vector< std::vector<int> > & vertexEdgeAdjecencyList
);

void createFaceFaceAdjacencyList(
        const Eigen::MatrixXi & F,
        std::vector< std::vector<int> > & faceFaceAdjecencyList
);

int adjacentFaceToEdge(
        const int v1,
        const int v2,
        const int old_face,
        const std::vector< std::vector<int> > & vertexFaceAdjecencyList
);

void adjacentFacesToEdge(
        const int v1,
        const int v2,
        const std::vector< std::vector<int> > & vertexFaceAdjecencyList,
        std::pair<int, int> & faces
);

int adjacentFaceToVertices(
        const int v1,
        const int v2,
        const int v3,
        const std::vector< std::vector<int> > & vertexFaceAdjecencyList
);

bool isBoundaryVertex(
        const Eigen::MatrixXd & V,
        int v,
        const std::vector< std::vector<int> > & vvAdj,
        const std::vector< std::vector<int> > &vfAdj
);

int edgeBetweenVertices(
        int v1,
        int v2,
        const std::vector< std::vector<int> > &veAdj
);

#endif //MESHMOULD_ADJACENCY_H
