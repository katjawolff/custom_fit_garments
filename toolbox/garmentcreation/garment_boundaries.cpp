#include "garment_boundaries.h"

#include "connectedSubmesh.h"
#include "../adjacency.h"
#include "dijkstra.h"
#include "remesh.h"

#include <array>

#include <igl/remove_unreferenced.h>
#include <igl/writeOFF.h>
#include <igl/adjacency_list.h>
#include <igl/cut_mesh.h>
#include <igl/facet_components.h>
#include <igl/boundary_loop.h>
#include <igl/signed_distance.h>

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<bool, Dynamic, Dynamic> MatrixXb;

GarmentBoundaries::GarmentBoundaries(MatrixXi & F)
: smooth(F)
{
    this->F = F;
    igl::adjacency_list(F, A);
}

int GarmentBoundaries::numberOfBoundaries() {
    return boundaries.size();
}

void GarmentBoundaries::addPointsToBoundary(const MatrixXd & V, int v_id) {
    // add the vertex to the list of clicked vertices
    if (activeBoundary.empty()) {
        activeBoundary.push_back(v_id);
    }
    // if we already have clicked vertices, find the shortest path to the last
    // and add all vertices in between
    else {
        int source = activeBoundary.back();

        VectorXd min_distance, previous;
        vector<int> path;
        //igl::dijkstra(source, target, A, min_distance, previous);		// this does not consider edge lengths 
		dijkstra(V, A, source, v_id, min_distance, previous);			// this does
        dijkstra(v_id, previous, path);

        for (int i = path.size() - 2; i >= 0; i--) {    // dont add the source vertex, since it's already in the list
            activeBoundary.push_back(path[i]);
        }
    }
}

void GarmentBoundaries::closeBoundary(const MatrixXd& V) {
	if (activeBoundary.empty())
		return;

    int source = activeBoundary.back();
    int target = activeBoundary.front();

    VectorXd min_distance, previous;
    vector<int> path;
	//igl::dijkstra(source, targets, A, min_distance, previous);	// this does not consider edge lengths 
	dijkstra(V, A, source, target, min_distance, previous);			// this does
    dijkstra(target, previous, path);

    for (int i = path.size() - 2; i >= 0; i--) {     // dont add the source, but the target to have it as the first and last entry in the list
        activeBoundary.push_back(path[i]);
    }

    // TODO: this is only necessary when geometry actually changes
    smooth.initGeometry(V);
    
    boundaries.push_back(smooth.smooth(activeBoundary, V, F));
    
    activeBoundary.clear();
}

void GarmentBoundaries::endBoundary(const Eigen::MatrixXd& V) { 
    boundaries.push_back(smooth.smooth(activeBoundary, V, F));
    activeBoundary.clear();
}

void GarmentBoundaries::deleteLast(){
    activeBoundary.clear();
    boundaries.pop_back();
}

void GarmentBoundaries::deleteAll(){
    activeBoundary.clear();
    boundaries.clear();
}

void GarmentBoundaries::saveBoundaries(const std::string fname)
{
    std::ofstream file(fname);
    
    file << boundaries.size() << " ";
    
    for(auto& b : boundaries)
    {
        file << b.size() << " ";
    }
    
    for(auto& b : boundaries)
    {
        for(auto p : b)
            file << p.first << " " << p.second(0) << " " << p.second(1) << " " << p.second(2) << "\n";
    }
    
    file.close();
}

void GarmentBoundaries::loadBoundaries(const std::string fname)
{
    std::ifstream file(fname);
    boundaries.clear();
    
    int len;
    file >> len;
    
    std::vector<int> blens;
    
    for(int j = 0; j < len; ++j)
    {
        int blen;
        file >> blen;
        blens.push_back(blen);
    }
    
    for(int j = 0; j < len; ++j)
    {
        std::vector<std::pair<int, Eigen::Vector3d>> b;
        
        for(int i = 0; i < blens[j]; ++i)
        {
            int id;
            double x, y, z;
            
            file >> id;
            file >> x;
            file >> y;
            file >> z;
            
            b.push_back(make_pair(id, Eigen::Vector3d(x, y, z)));
        }
        
        boundaries.push_back(b);
    }
    
    file.close();
}

void GarmentBoundaries::garmentFromBoundaries(const MatrixXd& V, int v_start, MatrixXd& Vg, MatrixXi& Fg, double refinement) {
    vector<bool> visited(V.rows(), false);
    garmentFromBoundaries(V, v_start, Vg, Fg, visited, refinement);
}

void GarmentBoundaries::garmentFromBoundaries(const MatrixXd& V, int v_start, MatrixXd& Vg, MatrixXi& Fg, vector<bool> & visited, double refinement) {
    
    // find new vertices
    std::map<std::pair<int, int>, int> vertexMap;
    std::vector<Eigen::RowVector3d> newVertices;
    
    std::vector<char> flag(V.rows(), -1);
    
    // debug write boundaries
  //  if(0) saveBoundaries("../boundaries");
  //  else loadBoundaries("../boundaries");
    
    for(auto& b : boundaries)
    {
        for(auto& v : b)
        {
            int eid = v.second[0] == .0 ? 0 : (v.second[1] == .0 ? 1 : 2);
        
            int i = F(v.first, (eid+1)%3);
            int j = F(v.first, (eid+2)%3);
            
            if( i < j )
            {
                vertexMap[make_pair(i,j)] = newVertices.size();
                newVertices.push_back( v.second[(eid+1)%3] * V.row(i) + v.second[(eid+2)%3] * V.row(j) );
            }
        }
    }
    
    // DFS
    std::queue<int> qu;
    qu.push(v_start);
   
    auto getVertexId = [&](const int i, const int j)
    {
        auto it = vertexMap.find( std::make_pair(std::min(i, j), std::max(i, j)) );
        if( it == vertexMap.end() ) return -1;
        else return it->second + (int)V.rows();
    };
    
    while(!qu.empty())
    {
        int i = qu.front();
        qu.pop();
         
        for(int j : A[i])
        {
            if(flag[j] == -1 && getVertexId(i, j) == -1)
            {
                flag[j] = 1;
                visited[j] = true;
                qu.push(j);
            }
        }
    }
    
    // split triangles
    std::vector<std::array<int, 3>> fNew;
    
    for(int i = 0; i < F.rows(); ++i)
    {
        std::array<int, 3> fl{ flag[F(i, 0)], flag[F(i, 1)], flag[F(i, 2)] };
        const char sum = fl[0] + fl[1] + fl[2];

        if(sum == -3) continue;
        
        if(sum == 3)
        {
            fNew.push_back({F(i, 0), F(i, 1), F(i, 2)});
        } else
        {
         
            Eigen::RowVector3i f = F.row(i);
            
            if(sum == 1) // two triangle case
            {
                const int id = fl[0] == -1 ? 0 : (fl[1] == -1 ? 1 : 2);
                const int a = getVertexId(f[id], f[(id+1) % 3]);
                const int b = getVertexId(f[id], f[(id+2) % 3]);
                
                fNew.push_back({f[(id+1) % 3], b, a});
                fNew.push_back({f[(id+1) % 3], f[(id+2) % 3], b});
                
            } else // one triangle case (sum == -1)
            {
                const int id = fl[0] == 1 ? 0 : (fl[1] == 1 ? 1 : 2);
                const int a = getVertexId(f[id], f[(id+1) % 3]);
                const int b = getVertexId(f[id], f[(id+2) % 3]);
                
                fNew.push_back({f[id], a, b});
            }
        }
    }
    
    
    // build final mesh
    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;
    
    V2.resize(V.rows() + newVertices.size(), 3);
    V2.topRows(V.rows()) = V;
    
    for(int i = 0; i < newVertices.size(); ++i)
    {
        V2.row(V.rows() + i) = newVertices[i];
    }
    
    F2.resize(fNew.size(), 3);
    for(int i = 0; i < fNew.size(); ++i)
    {
        F2(i, 0) = fNew[i][0];
        F2(i, 1) = fNew[i][1];
        F2(i, 2) = fNew[i][2];
    }
    
    Eigen::MatrixXd V3;
    Eigen::MatrixXi F3;
    Eigen::VectorXi I;
    igl::remove_unreferenced(V2, F2, V3, F3, I);
    
    if(refinement != 1.)
        remesh(V3, F3, Vg, Fg, refinement);
    else {
        Vg = V3;
        Fg = F3;
    }    

    // add a small offset for nicer rendering
    igl::AABB<Eigen::MatrixXd, 3> tree_col;		// collision AABB tree for the mannequin
    Eigen::MatrixXd FN_col, VN_col, EN_col;	    // vertices of the collision mesh
    Eigen::MatrixXi E_col;					    // triangles = faces of the garment mesh / faces of the collision mesh
    Eigen::VectorXi EMAP_col;
    tree_col.init(V, F);
    igl::per_face_normals(V, F, FN_col);
    igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN_col, VN_col);
    igl::per_edge_normals(V, F, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN_col, EN_col, E_col, EMAP_col);
    VectorXd S;
    VectorXi In;
    MatrixXd C, N;
    igl::signed_distance_pseudonormal(Vg, V, F, tree_col, FN_col, VN_col, EN_col, EMAP_col, S, In, C, N);

    double EPS = 0.003;
    for (int v = 0; v < Vg.rows(); v++) {
        if (S(v) < EPS) {
            Vector3d normal = N.row(v).normalized();	// unfortunately N seems not to be correctly normalized sometimes
            Vg.row(v) += (-S(v) + EPS) * normal;		// adjust position
        }
    }


    // --
    createCorrespondences(V, Vg, Fg);
}

void GarmentBoundaries::createCorrespondences(const MatrixXd & V, MatrixXd & Vg, MatrixXi & Fg) {
    // create a correspondence between the garment boundary vertices and the defined boundaries
    vector< vector<int> > L;
    igl::boundary_loop(Fg, L);
    mesh_boundary_ids.clear();
    boundary_ids.clear();
    boundary_v_ids.clear();
    boundary_p.clear();

    for (auto& l : L)
        mesh_boundary_ids.insert(mesh_boundary_ids.end(), l.begin(), l.end());

    for (auto vid : mesh_boundary_ids) {
        int bid = 0;
        int bvid = 0;
        double p = 0;
        double dist = 10000;

        for (size_t b = 0; b < boundaries.size(); b++) {
            Eigen::MatrixXd bverts;
            smooth.getCoordinates(V, boundaries[b], bverts);

            for (size_t i = 0; i < bverts.rows() / 2; ++i) {
                // get the 3D edge
                Vector3d e1 = bverts.row(2 * i);
                Vector3d e2 = bverts.row(2 * i + 1);

                double dist1 = (Vg.row(vid).transpose() - e1).norm();
                double dist2 = (Vg.row(vid).transpose() - e2).norm();
                double dist_new = dist1 + dist2;
                if (dist_new < dist) {
                    dist = dist_new;
                    bid = b;
                    bvid = 2 * i;
                    p = dist1 / dist_new;
                }
            }
        }

        boundary_ids.push_back(bid);
        boundary_v_ids.push_back(bvid);
        boundary_p.push_back(p);
    }
}

void GarmentBoundaries::cutGarmentAlongBoundaries(const MatrixXd& Vg, vector<MatrixXd> & Vg_list, vector<MatrixXi> & Fg_list) { 
    Vg_list.clear();
    Fg_list.clear();

    vector<bool> visited(Vg.rows(), false);
    for (int i = 0; i <= boundaries.size(); i++) {
        // for each boundary, we split the garment in 2 pieces -> boundary.size() + 1 pieces

        int v_start = 0;
        while (visited[v_start]) v_start++;
        if (v_start >= Vg.rows()) return;

        MatrixXd V_out;
        MatrixXi F_out;
        garmentFromBoundaries(Vg, v_start, V_out, F_out, visited, 1.);
        Vg_list.push_back(V_out);
        Fg_list.push_back(F_out);
    }
}

void GarmentBoundaries::markClosestBoundaryAsFixed(const MatrixXd& Vm, Vector3d& v) {

    // get coordinates of the current boundaries
    double dist = 1e10;
    double boundary_id = 0;

    for (int i = 0; i < boundaries.size(); i++) {
        MatrixXd bverts;
        smooth.getCoordinates(Vm, boundaries[i], bverts);
        for (int j = 0; j < bverts.rows(); j++) {
            double new_dist = (bverts.row(j).transpose() - v).norm();
            if (new_dist < dist) {
                dist = new_dist;
                boundary_id = i;
            }
        }
    }
    
    // fix
    bool already_fixed = false;
    for (int i = 0; i < fixed.size(); i++)
        if (fixed[i] == boundary_id)
            already_fixed = true;
    if (!already_fixed) {
        fixed.push_back(boundary_id);
        cout << "Fixed boundary." << endl;
    }
    else {
        fixed.erase(remove(fixed.begin(), fixed.end(), boundary_id), fixed.end());
        cout << "Released boundary." << endl;
    }
}

void GarmentBoundaries::getGarmentBoundaryFixedVertexPositions(const MatrixXd& Vm, MatrixXd& Vb, vector<int>& bids){
    // get coordinates of the current boundaries
    std::vector<Eigen::MatrixXd> boundaryEdgeVerts;
    for (auto& b : boundaries)
    {
        MatrixXd bverts;
        smooth.getCoordinates(Vm, b, bverts);
        boundaryEdgeVerts.push_back(bverts);
    }
    
    // get the positions along the boundaries where the garment vertices should be located
    bids.clear();
    vector< Vector3d > verts;

    for (size_t i = 0; i < boundary_ids.size(); i++) {
        int bid = boundary_ids[i];

        if (find(fixed.begin(), fixed.end(), bid) != fixed.end()) { // if this boundary is fixed
            int vid = boundary_v_ids[i];
            int p = boundary_p[i];
            Vector3d x1 = boundaryEdgeVerts[bid].row(vid);
            Vector3d x2 = boundaryEdgeVerts[bid].row(vid);
            verts.push_back( (1 - p) * x1 + p * x2 );
            bids.push_back(mesh_boundary_ids[i]);
        }
    }

    Vb.resize(verts.size(), 3);
    for (size_t i = 0; i < verts.size(); i++) {
        Vb.row(i) = verts[i];
    }
}

void GarmentBoundaries::getCylindersAroundBoundaryEdges(const MatrixXd & V, double radius, MatrixXd & Vb, MatrixXi & Fb) {
	std::vector< MatrixXd > cylinders;

	// read a cylinder
	MatrixXd Vcyl;
	MatrixXi Fcyl;
	cylinder(Vcyl, Fcyl);

    std::vector<Eigen::MatrixXd> boundaryEdgeVerts;
    
    // add closed & smoothed boundaries
    for(auto& b : boundaries)
    {
        Eigen::MatrixXd bverts;
        smooth.getCoordinates(V, b, bverts);
        boundaryEdgeVerts.push_back(bverts);
    }
    
    // add current boundary
    if(!activeBoundary.empty())
    {
        Eigen::MatrixXd bverts(2 * (activeBoundary.size() - 1), 3);
        for(size_t i = 0; i < activeBoundary.size() - 1; ++i)
        {
            bverts.row(2 * i) = V.row(activeBoundary[i]);
            bverts.row(2 * i + 1) = V.row(activeBoundary[i + 1]);
        }
        
        boundaryEdgeVerts.push_back(bverts);
    }
    
	for(auto& edges : boundaryEdgeVerts)
    {
		for (size_t i = 0; i < edges.rows() / 2; ++i)
        {
			// get the 3D edge
			Vector3d e1 = edges.row(2 * i);
			Vector3d e2 = edges.row(2 * i + 1);

			// adjust the original cylinder
			MatrixXd Vcyl_new(Vcyl.rows(), 3); Vcyl_new = Vcyl.replicate(1, 1);

			// stretch to edge length
			double l = (e1 - e2).norm();
			Vcyl_new.col(1) = Vcyl_new.col(1) * l;

			// adjust radius
			Vcyl_new.col(0) = Vcyl_new.col(0) * radius;
			Vcyl_new.col(2) = Vcyl_new.col(2) * radius;

			// rotate to edge orientation
			Vector3d a = Vector3d(0, 1, 0);
			Vector3d c = (e2 - e1).normalized();
			Matrix3d R = Quaterniond().setFromTwoVectors(a, c).toRotationMatrix();
			Vcyl_new = Vcyl_new * R.transpose();

			// translate to edge
			Vcyl_new.rowwise() += e1.transpose();

			cylinders.push_back(Vcyl_new);
		}
	}

	Vb.resize(Vcyl.rows() * cylinders.size(), 3);
	Fb.resize(Fcyl.rows() * cylinders.size(), 3);

	for (int i = 0; i < cylinders.size(); i++) {
		Vb.block(Vcyl.rows() * i, 0, Vcyl.rows(), 3) = cylinders[i];
		Fb.block(Fcyl.rows() * i, 0, Fcyl.rows(), 3) = Fcyl;
		Fcyl.array() += Vcyl.rows();
	}
}

void GarmentBoundaries::cylinder(MatrixXd & Vcyl, MatrixXi & Fcyl) {
	Vcyl.resize(32, 3);
	Fcyl.resize(32, 3);

	Vcyl <<
		1.000000, 0, 0.000000,
		0.923880, 0, 0.382683,
		0.707107, 0, 0.707107,
		0.382683, 0, 0.923880,
		0.000000, 0, 1.000000,
		-0.382683, 0, 0.923880,
		-0.707107, 0, 0.707107,
		-0.923880, 0, 0.382683,
		-1.000000, 0, 0.000000,
		-0.923880, 0, -0.382683,
		-0.707107, 0, -0.707107,
		-0.382683, 0, -0.923880,
		-0.000000, 0, -1.000000,
		0.382683, 0, -0.923880,
		0.707107, 0, -0.707107,
		0.923880, 0, -0.382683,
		1.000000, 1, 0.000000,
		0.923880, 1, 0.382683,
		0.707107, 1, 0.707107,
		0.382683, 1, 0.923880,
		0.000000, 1, 1.000000,
		-0.382683, 1, 0.923880,
		-0.707107, 1, 0.707107,
		-0.923880, 1, 0.382683,
		-1.000000, 1, 0.000000,
		-0.923880, 1, -0.382683,
		-0.707107, 1, -0.707107,
		-0.382683, 1, -0.923880,
		-0.000000, 1, -1.000000,
		0.382683, 1, -0.923880,
		0.707107, 1, -0.707107,
		0.923880, 1, -0.382683;

	Fcyl <<
		0, 16, 17,
		0, 17, 1,
		1, 17, 18,
		1, 18, 2,
		2, 18, 19,
		2, 19, 3,
		3, 19, 20,
		3, 20, 4,
		4, 20, 21,
		4, 21, 5,
		5, 21, 22,
		5, 22, 6,
		6, 22, 23,
		6, 23, 7,
		7, 23, 24,
		7, 24, 8,
		8, 24, 25,
		8, 25, 9,
		9, 25, 26,
		9, 26, 10,
		10, 26, 27,
		10, 27, 11,
		11, 27, 28,
		11, 28, 12,
		12, 28, 29,
		12, 29, 13,
		13, 29, 30,
		13, 30, 14,
		14, 30, 31,
		14, 31, 15,
		15, 31, 16,
		15, 16, 0;
}
