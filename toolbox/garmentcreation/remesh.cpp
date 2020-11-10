#include "remesh.h"

#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/SurfaceRemeshing.h>
#include <igl/avg_edge_length.h>
#include <igl/writeOFF.h>

void remesh(const Eigen::MatrixXd& Vin, const Eigen::MatrixXi& Fin,
            Eigen::MatrixXd& Vout, Eigen::MatrixXi& Fout,
            const double refinementFactor)
{
    assert(refinementFactor != .0);
    
    pmp::SurfaceMesh mesh;
    
    const int nv = (int)Vin.rows();
    const int nf = (int)Fin.rows();
    
    // to surface mesh
    std::vector<pmp::Vertex> vertices(nv);
    
    for(int i = 0; i < nv; ++i)
        vertices[i] = mesh.add_vertex(pmp::Point(Vin(i,0), Vin(i,1), Vin(i,2)));
    
    for(int i = 0; i < nf; ++i)
        mesh.add_triangle(vertices[Fin(i, 0)], vertices[Fin(i, 1)], vertices[Fin(i, 2)]);
    
    // perform remeshing
    pmp::SurfaceRemeshing rm(mesh);
    const double edgeLength = igl::avg_edge_length(Vin, Fin);
    rm.uniform_remeshing(1. / refinementFactor * edgeLength, 10, false);
    
    // collapse edges small edges (= small relative to other edges in a triangle.)

    for(auto f : mesh.faces())
    {
        if(!f.is_valid()) continue;
        const pmp::Halfedge hes[3]
        {
            mesh.halfedge(f),
            mesh.next_halfedge(mesh.halfedge(f)),
            mesh.next_halfedge(mesh.next_halfedge(mesh.halfedge(f)))
        };
        
        double len[3]
        {
            mesh.edge_length(mesh.edge(hes[0])),
            mesh.edge_length(mesh.edge(hes[1])),
            mesh.edge_length(mesh.edge(hes[2]))
        };
    
        int mi;
                   
        if(len[0] < len[1] && len[0] < len[2]) mi = 0;
        else if(len[1] < len[0] && len[1] < len[2]) mi = 1;
        else mi = 2;
   
        int ma = len[(mi + 1) % 3] > len[(mi + 2) % 3] ? (mi + 1) % 3 : (mi + 2) % 3;
        
        const double r = len[ma] / len[mi];
        
        if(r > 3 && mesh.is_collapse_ok(hes[mi]))
        {
            mesh.collapse(hes[mi]);
        }
    }
    
    mesh.garbage_collection();
    
    // back to igl format
    Vout.resize(mesh.n_vertices(), 3);
    Fout.resize(mesh.n_faces(), 3);
    
    std::map<pmp::Vertex, int> idMap;
    
    int i = 0;
    for(auto v : mesh.vertices())
    {
        auto p = mesh.position(v);
        assert(i < mesh.n_vertices());
        
        for(int j = 0; j < 3; ++j)
            Vout(i, j) = p[j];

        idMap[v] = i;
        ++i;
    }
    
    assert(i == mesh.n_vertices());
    
    i = 0;
    
    for(auto f : mesh.faces())
    {
        auto he = mesh.halfedge(f);
        
        assert(i < mesh.n_faces());
        
        Fout(i, 0) = idMap[mesh.to_vertex(he)];
        he = mesh.next_halfedge(he);
    
        Fout(i, 1) = idMap[mesh.to_vertex(he)];
        he = mesh.next_halfedge(he);
        
        Fout(i, 2) = idMap[mesh.to_vertex(he)];

        ++i;
    }
}
