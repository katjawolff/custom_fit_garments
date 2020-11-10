#include "adjacency.h"

#include <igl/edges.h>

using namespace std;
using namespace Eigen;

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
void createFacePairEdgeListWith4VerticeIDs(
    const MatrixXi& F,
    const MatrixXi& E,
    const vector<vector<int>>& vf_adj,
    MatrixXi& E4,
    MatrixXi& EF6,
    MatrixXi& ef_adj
) {
    vector< vector<int> > E_4v;

    EF6.resize(E.rows(), 6);
    ef_adj.resize(E.rows(), 2);

    for (int e = 0; e < E.rows(); e++) {
        pair<int, int> faces;
        adjacentFacesToEdge(E(e, 0), E(e, 1), vf_adj, faces);

        // if the second face does not exist, swap the order, such that v2 is always adjacent to a face (always face.second)
        // i.e. if a second face does not exist, v3 will not exist either
        if (faces.second == -1)
            faces = make_pair(-1, faces.first);

        // store the adjacent face id's in the right order. The first always exists
        ef_adj.row(e) = RowVector2i(faces.second, faces.first);

        // create v0..v3 in the right order
        int v0, v1, v2, v3;

        if (F(faces.second, 0) == E(e, 0)) {
            if (F(faces.second, 1) == E(e, 1)) {
                v0 = E(e, 0);
                v1 = E(e, 1);
                v2 = F(faces.second, 2);
            }
            else {
                v0 = E(e, 1);
                v1 = E(e, 0);
                v2 = F(faces.second, 1);
            }
        }
        else if (F(faces.second, 1) == E(e, 0)) {
            if (F(faces.second, 0) == E(e, 1)) {
                v0 = E(e, 1);
                v1 = E(e, 0);
                v2 = F(faces.second, 2);
            }
            else {
                v0 = E(e, 0);
                v1 = E(e, 1);
                v2 = F(faces.second, 0);
            }
        }
        else {
            if (F(faces.second, 0) == E(e, 1)) {
                v0 = E(e, 0);
                v1 = E(e, 1);
                v2 = F(faces.second, 1);
            }
            else {
                v0 = E(e, 1);
                v1 = E(e, 0);
                v2 = F(faces.second, 0);
            }
        }

        v3 = -1;
        if (faces.first != -1) {
            if (F(faces.first, 0) != E(e, 0) && F(faces.first, 0) != E(e, 1))
                v3 = F(faces.first, 0);
            else if (F(faces.first, 1) != E(e, 0) && F(faces.first, 1) != E(e, 1))
                v3 = F(faces.first, 1);
            else if (F(faces.first, 2) != E(e, 0) && F(faces.first, 2) != E(e, 1))
                v3 = F(faces.first, 2);
        }

        E_4v.push_back({ v2,v0,v1,v3 });
    }

    E4.resize(E_4v.size(), 4);
    for (int e = 0; e < E_4v.size(); e++) {
        E4.row(e) << E_4v[e][0], E_4v[e][1], E_4v[e][2], E_4v[e][3];
    }

    // indexing for EF6
    for (int e = 0; e < E.rows(); e++) {
        int f = ef_adj(e, 0);
        int f_dot = ef_adj(e, 1);

        for (int i = 0; i < 3; i++) {       // go over vertices in the face
            for (int j = 0; j < 3; j++) {   // go over vertices in the edge+2faces
                if (E4(e, j) == F(f, i)) EF6(e, j) = i;
                if (f_dot != -1 && E4(e, j + 1) == F(f_dot, i)) EF6(e, j + 3) = i;
            }
        }
    }
}

// for each non-boundary edge, we collect all 4 corresponding vertex ids in a specific order
// this one is for the quadratic bending energy
void createFacePairEdgeListWith4VerticeIDs(
    const MatrixXi& F,
    MatrixXi& E4
) {
    MatrixXi E;
    vector< vector<int> > vf_adj;
    vector< vector<int> > E_4v;

    igl::edges(F, E);
    createVertexFaceAdjacencyList(F, vf_adj);

    for (int e = 0; e < E.rows(); e++) {
        pair<int, int> faces;
        adjacentFacesToEdge(E(e, 0), E(e, 1), vf_adj, faces);

        if (faces.first != -1 && faces.second != -1) {
            int v0, v1, v2, v3;

            if (F(faces.second, 0) == E(e, 0)) {
                if (F(faces.second, 1) == E(e, 1)) {
                    v0 = E(e, 0);
                    v1 = E(e, 1);
                    v2 = F(faces.second, 2);
                }
                else {
                    v0 = E(e, 1);
                    v1 = E(e, 0);
                    v2 = F(faces.second, 1);
                }
            }
            else if (F(faces.second, 1) == E(e, 0)) {
                if (F(faces.second, 0) == E(e, 1)) {
                    v0 = E(e, 1);
                    v1 = E(e, 0);
                    v2 = F(faces.second, 2);
                }
                else {
                    v0 = E(e, 0);
                    v1 = E(e, 1);
                    v2 = F(faces.second, 0);
                }
            }
            else {
                if (F(faces.second, 0) == E(e, 1)) {
                    v0 = E(e, 0);
                    v1 = E(e, 1);
                    v2 = F(faces.second, 1);
                }
                else {
                    v0 = E(e, 1);
                    v1 = E(e, 0);
                    v2 = F(faces.second, 0);
                }
            }

            if (F(faces.first, 0) != E(e, 0) && F(faces.first, 0) != E(e, 1))
                v3 = F(faces.first, 0);
            else if (F(faces.first, 1) != E(e, 0) && F(faces.first, 1) != E(e, 1))
                v3 = F(faces.first, 1);
            else if (F(faces.first, 2) != E(e, 0) && F(faces.first, 2) != E(e, 1))
                v3 = F(faces.first, 2);

            E_4v.push_back({ v0,v1,v2,v3 });
        }
    }
    E4.resize(E_4v.size(), 4);
    for (int e = 0; e < E_4v.size(); e++) {
        E4.row(e) << E_4v[e][0], E_4v[e][1], E_4v[e][2], E_4v[e][3];
    }
}

// creates an adjacency list, for each vertex, adjacent faces are listed
void createVertexFaceAdjacencyList(
    const MatrixXi& F,
    vector<vector<int>>& vertexFaceAdjecencyList
) {
    vertexFaceAdjecencyList.clear();
    int number_of_verts = F.maxCoeff() + 1;

    vertexFaceAdjecencyList.resize(number_of_verts);

    for (int i = 0; i < F.rows(); i++) {         // loop through faces
        for (int j = 0; j < F.cols(); j++) {     // loop through face's vertices
            vertexFaceAdjecencyList[F(i, j)].push_back(i); // store face index for all these vertices
        }
    }
}

// for each face, store neighbouring edge IDs
void createFaceEdgeAdjecencyList(
    const MatrixXi& F,
    const MatrixXi& E,
    const vector<vector<int>>& vf_adj,
    vector<vector<int>>& fe_adj
) {
    fe_adj.clear();
    fe_adj.resize(F.rows());

    // get all adjacent edges to each face
    for (int i = 0; i < E.rows(); i++) {         // loop through edges
        pair<int, int> faces;
        adjacentFacesToEdge(E(i, 0), E(i, 1), vf_adj, faces);

        if (faces.first != -1)
            fe_adj[faces.first].push_back(i); // store face index for all these vertices
        if (faces.second != -1)
            fe_adj[faces.second].push_back(i);
    }

    // sort s.t. edges are clockwise opposites to v0,v1,v2
    for (int f = 0; f < F.rows(); f++) {
        vector<int> e(3);

        for (int i = 0; i < 3; i++) {
            // for each of the three edge ids: check at which position they need to be
            int e_id = fe_adj[f][i];

            for (int j = 0; j < 3; j++) {
                // go through each face edge and check if that corresponds to the edge id
                if (E(e_id, 0) == F(f, j)) {            // e.g. start = 0
                    if (E(e_id, 1) == F(f, (j + 1) % 3))    // end = 1
                        e[(j + 2) % 3] = e_id;            // opposite = 2
                    else
                        e[(j + 1) % 3] = e_id;            // end = 2, opposite = 1
                }
            }
        }
        for (int i = 0; i < 3; i++)
            fe_adj[f][i] = e[i];
    }
}


// for each face, store neighbouring edge IDs
void createVertexEdgeAdjecencyList(
    const MatrixXi& E,
    vector<vector<int>>& vertexEdgeAdjecencyList
) {
    int number_of_verts = E.maxCoeff() + 1;

    vertexEdgeAdjecencyList.clear();
    vertexEdgeAdjecencyList.resize(number_of_verts);

    for (int i = 0; i < E.rows(); i++) {         // loop through edges
        vertexEdgeAdjecencyList[E(i, 0)].push_back(i);
        vertexEdgeAdjecencyList[E(i, 1)].push_back(i);
    }
}

int adjacentFaceToEdge(
    const int v1,
    const int v2,
    const int old_face,
    const vector<vector<int>>& vertexFaceAdjecencyList
) {
    vector<int> ring1 = vertexFaceAdjecencyList[v1];
    vector<int> ring2 = vertexFaceAdjecencyList[v2];

    for (int i = 0; i < ring1.size(); i++) {
        for (int j = 0; j < ring2.size(); j++) {
            if (ring1[i] == ring2[j] && ring1[i] != old_face)
                return ring1[i];
        }
    }

    return -1;
}

int adjacentFaceToVertices(
    const int v1,
    const int v2,
    const int v3,
    const vector<vector<int>>& vertexFaceAdjecencyList
) {
    vector<int> ring1 = vertexFaceAdjecencyList[v1];
    vector<int> ring2 = vertexFaceAdjecencyList[v2];
    vector<int> ring3 = vertexFaceAdjecencyList[v3];

    for (int i = 0; i < ring1.size(); i++) {
        for (int j = 0; j < ring2.size(); j++) {
            if (ring1[i] == ring2[j] && ring1[i] != -1) {
                for (int k = 0; k < ring3.size(); k++) {
                    if (ring1[i] == ring3[k])
                        return ring1[i];
                }
            }
        }
    }

    return -1;
}

void adjacentFacesToEdge(
    const int v1,
    const int v2,
    const vector<vector<int>>& vertexFaceAdjecencyList,
    pair<int, int>& faces
) {
    vector<int> ring1 = vertexFaceAdjecencyList[v1];
    vector<int> ring2 = vertexFaceAdjecencyList[v2];
    bool found_first = false;

    faces.first = -1;
    faces.second = -1;

    for (int i = 0; i < ring1.size(); i++) {
        for (int j = 0; j < ring2.size(); j++) {
            if (ring1[i] == ring2[j] && !found_first) {
                faces.first = ring1[i];
                found_first = true;
            }
            else if (ring1[i] == ring2[j] && found_first)
                faces.second = ring1[i];
        }
    }
}

void createFaceFaceAdjacencyList(
    const MatrixXi& F,
    vector<vector<int>>& faceFaceAdjecencyList
) {
    faceFaceAdjecencyList.clear();
    faceFaceAdjecencyList.resize(F.rows());

    vector<vector<int>> vfAdj;
    createVertexFaceAdjacencyList(F, vfAdj);

    for (int f = 0; f < F.rows(); f++) {
        for (int e = 0; e < 3; e++) {
            pair<int, int> faces;
            adjacentFacesToEdge(F(f, e), F(f, (e + 1) % 3), vfAdj, faces);

            if (faces.first == f && faces.second != -1) faceFaceAdjecencyList[f].push_back(faces.second);
            if (faces.second == f && faces.first != -1) faceFaceAdjecencyList[f].push_back(faces.first);
        }
    }
}

bool isBoundaryVertex(
    const MatrixXd& V,
    int v,
    const vector< vector<int> >& vvAdj,
    const vector< vector<int> >& vfAdj
) {
    for (int i = 0; i < vvAdj[v].size(); i++) {
        pair<int, int> faces;
        adjacentFacesToEdge(v, vvAdj[v][i], vfAdj, faces);
        if (faces.first == -1 || faces.second == -1)
            return true;
    }

    return false;
}

int edgeBetweenVertices(
    int v1,
    int v2,
    const vector< vector<int> >& veAdj
) {
    vector<int> edgeRingV1 = veAdj[v1];
    vector<int> edgeRingV2 = veAdj[v2];

    for (int i = 0; i < (int)edgeRingV1.size(); i++)
        for (int j = 0; j < (int)edgeRingV2.size(); j++)
            if (edgeRingV1[i] == edgeRingV2[j])
                return edgeRingV1[i];

    return -1;
}
