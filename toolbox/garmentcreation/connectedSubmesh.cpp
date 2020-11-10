#include "connectedSubmesh.h"

#include <vector>
#include <igl/facet_components.h>

using namespace std;
using namespace Eigen;

void getBiggestConnectedSubmesh(
	const MatrixXd& V_in,
	const MatrixXi& F_in,
	MatrixXd& V_sub,
	MatrixXi& F_sub
) {
	// get the face ids of the connected components
	VectorXi ids;
	igl::facet_components(F_in, ids);

	// get the id of the component with the most vertices
	int c = ids.maxCoeff();
	vector< int > count(c + 1, 0);
	for (int i = 0; i < ids.rows(); i++)
		count[ids(i)]++;
	int component = -1;
	int vert_number = 0;
	for (int i = 0; i < count.size(); i++) {
		if (count[i] > vert_number) {
			component = i;
			vert_number = count[i];
		}
	}

	// now extract that component
	getSubmesh(V_in, F_in, ids, component, V_sub, F_sub);
}

void getSubmesh(
	const MatrixXd& V_in, 
	const MatrixXi& F_in, 
	const VectorXi& ids, 
	int component, 
	MatrixXd& V_sub, 
	MatrixXi& F_sub
) {
	// this works, because we know that we are getting a whole connected submesh
	// i.e. if a face gets removed, all its vertices get removed too
	vector< Vector3d > vertex_list;
	vector< Vector3i > face_list;
	vector< int > new_vertex_id(V_in.rows(), -1);

	// collect all vertices and faces of the submesh
	for (int f = 0; f < ids.rows(); f++) {
		if (ids[f] == component) {
			// keep this face and the vertices
			Vector3i new_face;

			for (int i = 0; i < 3; i++) {
				int v_old = F_in(f, i);
				int v_new = new_vertex_id[v_old];			// get the vertex index, if we already added it
				if (v_new == -1) {
					v_new = vertex_list.size();				// add this new vertex to our list and give it a new index
					new_vertex_id[v_old] = v_new;			// remember the new indey
					vertex_list.push_back(V_in.row(v_old));	// add the vertex coordinates
				}
				new_face(i) = v_new;						// write the new index into the face
			}

			face_list.push_back(new_face);
		}
	}

	// write into a matrix
	V_sub.resize(vertex_list.size(), 3);
	F_sub.resize(face_list.size(), 3);

	for (int v = 0; v < vertex_list.size(); v++)
		V_sub.row(v) = vertex_list[v];
	for (int f = 0; f < face_list.size(); f++)
		F_sub.row(f) = face_list[f];
}