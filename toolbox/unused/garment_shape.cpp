#include "garment_shape.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <igl/edges.h>
#include "../adjacency.h"

using namespace std;
using namespace Eigen;

typedef Triplet<double> Tri;


RestShape::RestShape(){}

void RestShape::init(const MatrixXi& F) {
    // create adjacency information - E4: list of 4 vertices for each face pair (each internal edge)
    this->F = F;
    igl::edges(F, E);
    vector< vector<int> > VF_adj;
    createVertexFaceAdjacencyList(F, VF_adj);
    createFaceEdgeAdjecencyList(F, E, VF_adj, FE_adj);
    createFacePairEdgeListWith4VerticeIDs(F, E, VF_adj, E4, EF6, EF_adj);
    faces = F.rows();
    edges = E.rows();

    interior_edges = 0;
    for (int i = 0; i < EF_adj.rows(); i++)
        if (EF_adj(i, 1) != -1)
            interior_edges++;
}

void RestShape::setFixedVertices(const vector<bool>& fixed_boundary_vert) {
    this->fixed_boundary_vert = fixed_boundary_vert;

    fixed_boundary_edge.clear();
    fixed_boundary_edge.resize(E.rows(), false);
    for (int e = 0; e < E.rows(); e++) 
        if (fixed_boundary_vert[E(e, 0)] && fixed_boundary_vert[E(e, 1)])
            fixed_boundary_edge[e] = true;
}

void RestShape::shape_energy(
    const MatrixXd& X_current,
    const MatrixXd& X_restshape,
    const vector< double >& length_target,
    const vector< double >& angle_target,
    int edge_constraints_count,
    const vector< int >& edge_constraints,
    VectorXd& f,
    SparseMatrix<double>& J,
    bool with_jacobian
) {
    // edge lenght weight is always 1.
    double alpha = 1.e-4;       // how to weight angle constraints
    //double beta = 1.e-6;        // how to weight shear constraints
    double lambda = 1.e-4;      // how to weight positional constraints
    //double lambda_fixed = 1.;
    double gamma = 1e3;       // weighing fixed edge lengths of fixed boundaries

    // calculate energy and Jacobian
    // -----------------------------
    int total_constraints = edges + interior_edges /*+ faces*/ + verts * 3 + 1; // edge constr. + angle constr. + shear constr. + position constr. + sum edges
    // TODO abobve line
    f = VectorXd::Zero(total_constraints);
    int c = 0; //index of next constraint    
    vector<Tri> triJ;
    triJ.reserve(edges * 6 + interior_edges * 12 + verts * 3 + faces * 3);
    // TODO abobve line

    // edge lengths 
    // ------------
    vector<bool> vertex_constrained_by_edge(X_current.rows(), false);
    VectorXd edge_length(edges);
    for (int e = 0; e < edges; e++) {
        int x1 = E(e, 0);
        int x2 = E(e, 1);

        if (edge_constraints[e] != 0) { // mark vertices that are affected by edge length changes
            vertex_constrained_by_edge[x1] = true;
            vertex_constrained_by_edge[x2] = true;
        }

        Vector3d dx = X_current.row(x1) - X_current.row(x2);
        edge_length(e) = dx.norm();

        if (!fixed_boundary_edge[e]) {  // TODO try including the boundary edges too
            f(c) = dx.dot(dx) - length_target[e] * length_target[e];

            if (with_jacobian) {
                Vector3d df_dx = 2. * dx;    // df_dx2 = -df_dx1
                for (int j = 0; j < 3; j++) {
                    triJ.push_back(Tri(c, j * verts + x1, df_dx(j)));
                    triJ.push_back(Tri(c, j * verts + x2, -df_dx(j)));
                }
            }
            c++;
        }
    }

    // fixed boundary edges
    // =====================
    // TODO !!! need to do this for each boundary individually!!
/*    double sum_old = 0.;
    double sum_new = 0.; 

    for (int e = 0; e < edges; e++) {
        if (fixed_boundary_edge[e]) {
            int x1 = E(e, 0);
            int x2 = E(e, 1);
            sum_new += edge_length(e);
            sum_old += (X_restshape.row(x1) - X_restshape.row(x2)).norm();
        }
    }
    double d_sum = sum_new - sum_old;
    f(c) = gamma * d_sum * d_sum;

    for (int e = 0; e < edges; e++) {
        if (fixed_boundary_edge[e]) {
            int x1 = E(e, 0);
            int x2 = E(e, 1);
            Vector3d dx = X_current.row(x1) - X_current.row(x2);

            if (with_jacobian) {
                Vector3d df_dx = gamma * 2. * d_sum * dx / edge_length(e);
                for (int j = 0; j < 3; j++) {
                    triJ.push_back(Tri(c, j * verts + x1, df_dx(j)));
                    triJ.push_back(Tri(c, j * verts + x2, -df_dx(j)));
                }
            }
        }
    }
    c++;
    */
    // shear
    // -----  
 /*   for (int i = 0; i < faces; i++) {        
        for (int k = 0; k < 3; k++) {   // each angle in a triangle
            Vector3d v1 = X_current.row(F(i, (k+1)%3)) - X_current.row(F(i, k));
            Vector3d v2 = X_current.row(F(i, (k+2)%3)) - X_current.row(F(i, k));
            double v1norm = v1.norm();
            double v2norm = v2.norm();
            double shear = v1.dot(v2) / (v1norm * v2norm);
            if (shear > 0.9) {
                f(c) = shear - 0.9;
                c++;

                // first order derivatives
                // some mysterious needed precomp (for dJ/dv_x), left col for J_x, right col for J_y
                if (with_jacobian) {

                    Vector3d df_dv1 = v2 / (v1norm * v2norm) * (v1.sum() / (v1norm * v1norm) - 1);
                    Vector3d df_dv2 = v1 / (v1norm * v2norm) * (v2.sum() / (v2norm * v2norm) - 1);
                    Vector3d df_dv0 = v2 / (v1norm * v2norm) * (-v1.sum() / (v1norm * v1norm) + 1) + v1 / (v1norm * v2norm) * (-v2.sum() / (v2norm * v2norm) + 1);
                    for (int j = 0; j < 3; j++) {   // x,y,z           
                        triJ.push_back(Tri(c, j * verts + F(i, 0), df_dv0(j)));
                        triJ.push_back(Tri(c, j * verts + F(i, 1), df_dv1(j)));
                        triJ.push_back(Tri(c, j * verts + F(i, 2), df_dv2(j)));
                    }
                }
            }
        }
    }
*/
    // angles
    // ------
    // to avoid wobbling and overlapping triangles
    MatrixXd normal(faces, 3), cos_alpha(faces, 3), h_inverse(faces, 3);
    for (int f = 0; f < faces; f++) {
        vector<Vector3d> e(3);
        e[0] = X_current.row(F(f, 2)) - X_current.row(F(f, 1));		// edge opposite of v_0
        e[1] = X_current.row(F(f, 0)) - X_current.row(F(f, 2));		// opposite of v_1
        e[2] = X_current.row(F(f, 1)) - X_current.row(F(f, 0));		// opposite of v_2

        normal.row(f) = e[2].cross(-e[1]).normalized();

        if (with_jacobian) {
            double area = 0.5 * e[0].cross(-e[2]).norm();

            vector<Vector3d> en(3);
            for (int i = 0; i < 3; i++)
                en[i] = e[i].normalized();			// direction is important, therefore we do not precompute them per edge earlier

            vector<Vector3d> edge_normal(3);
            for (int i = 0; i < 3; i++) {
                cos_alpha(f, i) = -en[(i + 1) % 3].dot(en[(i + 2) % 3]);	// angle at v_i
                h_inverse(f, i) = 0.5 * edge_length(FE_adj[f][i]) / area;	// height ending in v_i
            }
        }
    }

    for (int e = 0; e < edges; e++) {
        if (EF_adj(e, 1) != -1) { // no border edge

            // indexing
            int face = EF_adj(e, 0);		// adjacent face 1
            int face_dot = EF_adj(e, 1);	// adjacent face 2

            // angle
            double sin_theta_half = (normal.row(face) - normal.row(face_dot)).norm() * 0.5;
            double cos_theta_half = (normal.row(face) + normal.row(face_dot)).norm() * 0.5;
            double theta = atan2(sin_theta_half, cos_theta_half) * 2.;

            f(c) = alpha * (theta - angle_target[e]);

            // first derivative 
            if (with_jacobian) {
                // indexing
                vector<int> f_v(3), f_dot_v(3);	// f_v 0,1,2 corresponds to v0,v1,v2 --- f_dot_v 0,1,2 corresponds to v1,v2,v3
                for (int i = 0; i < 3; i++) {
                    f_v[i] = EF6(e, i);
                    f_dot_v[i] = EF6(e, 3 + i);
                }

                // jacobian
                vector< Vector3d > delta_theta(4);
                delta_theta[0] = -h_inverse(face, f_v[0]) * normal.row(face);
                delta_theta[1] = cos_alpha(face, f_v[2]) * h_inverse(face, f_v[1]) * normal.row(face) + cos_alpha(face_dot, f_dot_v[1]) * h_inverse(face_dot, f_dot_v[0]) * normal.row(face_dot);
                delta_theta[2] = cos_alpha(face, f_v[1]) * h_inverse(face, f_v[2]) * normal.row(face) + cos_alpha(face_dot, f_dot_v[0]) * h_inverse(face_dot, f_dot_v[1]) * normal.row(face_dot);
                delta_theta[3] = -h_inverse(face_dot, f_dot_v[2]) * normal.row(face_dot);

                for (int i = 0; i < 4; i++)       // x0,x1,x2,x3
                    for (int j = 0; j < 3; j++)   // x,y,z
                        triJ.push_back(Tri(c, j * verts + E4(e, i), alpha * delta_theta[i](j)));
            }

            c++;
        }
    }

    // positions
    // ---------
    // constrain each vertex that hasn't been affected
    // and those that are on fixed boundaries
    for (int v = 0; v < verts; v++) {
        for (int i = 0; i < 3; i++) {
            if (!vertex_constrained_by_edge[v] /*&& !fixed_boundary_vert[v]*/) {
                f(c) = lambda * (X_current(v, i) - X_restshape(v, i));
                if (with_jacobian)
                    triJ.push_back(Tri(c, i * verts + v, lambda));
                c++;
            }
        }
    }

    if (with_jacobian) {
        J.resize(total_constraints, verts * 3);
        J.setFromTriplets(triJ.begin(), triJ.end());
    }
}

void RestShape::adjust_restshape(
    const MatrixXd& X_simulation,
    const MatrixXd& Vm,
    const double t_stretch,
    const double t_compress,
    MatrixXd& X_restshape
) {
    vector<double> edge_adjustment(edges, 0.);
    adjust_restshape(X_simulation, Vm, t_stretch, t_compress, edge_adjustment, X_restshape);
}

void RestShape::adjust_restshape(
    const MatrixXd& X_simulation,
    const MatrixXd& Vm,
    const double t_stretch,
    const double t_compress,
    const vector<double>& edge_adjustment,
    MatrixXd & X_restshape
) {
    verts = X_restshape.rows();
    
    // calculate the new length of each edge
    // -------------------------------------
    vector< double > l_target(edges);
    double original_diff = 0;
    vector<int> edge_constraints(edges, 0);
    int edge_constraints_count = 0;
    for (int e = 0; e < edges; e++) {
        double l_rest = (X_restshape.row(E(e, 0)) - X_restshape.row(E(e, 1))).norm() + edge_adjustment[e];
        double l_sim  = (X_simulation.row(E(e, 0)) - X_simulation.row(E(e, 1))).norm();
        double ratio = l_sim / l_rest;
        
        if (ratio > t_stretch) {
            l_target[e] = l_sim / t_stretch;    // es soll: l_rest_neu * t_rtretch = l_sim
            edge_constraints[e] = edge_constraints_count;
            edge_constraints_count++;
            //cout << "-- edge " << e << " too short " << l_rest << " need " << l_target[e] << endl;
        }
        else if (ratio < t_compress) {
            l_target[e] = l_sim / t_compress;
            edge_constraints[e] = true;
            edge_constraints_count++;
            //cout << "-- edge " << e << " too long " << l_rest << " need " << l_target[e] << endl;
        }
        else {
            l_target[e] = l_rest;
        }
        original_diff += (l_target[e]* l_target[e] - l_rest* l_rest)* (l_target[e] * l_target[e] - l_rest * l_rest);
    }
    if (edge_constraints_count== 0) return;

    // calculate old angles of the rest shape
    // --------------------------------------
    vector< double > old_theta(edges, 0);
    MatrixXd old_normal(faces, 3);
    for (int f = 0; f < faces; f++) {
        Vector3d e1 = X_restshape.row(F(f, 2)) - X_restshape.row(F(f, 0));		// opposite of v_1
        Vector3d e2 = X_restshape.row(F(f, 1)) - X_restshape.row(F(f, 0));		// opposite of v_2
        old_normal.row(f) = e2.cross(e1).normalized();
    }

    for (int e = 0; e < edges; e++) {
        if (EF_adj(e, 1) != -1) { // no border edge

            // indexing
            int face = EF_adj(e, 0);		// adjacent face 1
            int face_dot = EF_adj(e, 1);	// adjacent face 2

            // angle
            double sin_theta_half = (old_normal.row(face) - old_normal.row(face_dot)).norm() * 0.5;
            double cos_theta_half = (old_normal.row(face) + old_normal.row(face_dot)).norm() * 0.5;
            old_theta[e] = atan2(sin_theta_half, cos_theta_half) * 2.;
            //old_theta[e] = 0.;
        }
    }

    // use Gauss-Newton to calculate the new rest shape with new edge lengths
    // ----------------------------------------------------------------------
    double residual = 1.;
    MatrixXd X_current = X_restshape;
    bool change = false;
    double eps = 0.;// 1.e-8;

    while (residual > eps) {   // repeat Gauss-Newton steps 

        VectorXd f;
        SparseMatrix<double> J;
        shape_energy(X_current, X_restshape, l_target, old_theta, edge_constraints_count, edge_constraints, f, J, true);

        double energy_start_of_step = f.dot(f);
        if (energy_start_of_step < 1e-8) {
            cout << "-- energy almost 0 already" << endl;
            break;     // nothing to do here
        }

        // find the step direction / solve system
        // -----------------------
        SparseMatrix<double> D = J.transpose() * J;
        VectorXd dir_vec;
        LDLTSolver solver;
        solver.setSystem(D);
        solver.solve(J.transpose() * f, dir_vec);

        MatrixXd dir(verts, 3);    // transform from vector to Matrix with x,y,z columns
        dir.col(0) = dir_vec.head(verts);
        dir.col(1) = dir_vec.segment(verts, verts);
        dir.col(2) = dir_vec.tail(verts);

        // find best step size
        // -------------------
        MatrixXd X_updated;
        MatrixXd X_current_stepsize = X_current;
        double stepsize = 0.0001;
        //double stepsize = 1.;
        // We start with a small step size, since most of the time we are already ultra close to the best solution. We double the stepsize when possible.
        // If the energy only gets larger, even for this tiny step size, we can break instantly and just use the old solution.  
        double energy_updated;
        double energy_current_stepsize = energy_start_of_step;
        do {
            energy_updated = energy_current_stepsize;
            X_updated = X_current_stepsize;
            stepsize *= 2.;

            X_current_stepsize = X_current - stepsize * dir; // new solution
            VectorXd f_step;
            shape_energy(X_current_stepsize, X_restshape, l_target, old_theta, edge_constraints_count, edge_constraints, f_step, J, false);   // new energy, no jacobian needed
            energy_current_stepsize = f_step.dot(f_step);

        } while (energy_current_stepsize < (energy_updated - eps) && stepsize < 100.);

        // check, if the energy went down
        residual = energy_start_of_step - energy_updated;
        if (residual <= eps) {  // no improvement, stop Gauss-Newton
            //X_updated = X_current;
            cout << "-- Gauss-Newton only made it worse." << endl;
            cout << "-- final stepsize " << stepsize << endl;
            cout << "-- energy old " << fixed << setprecision(16) << energy_start_of_step << endl;
            cout << "-- energy new " << energy_current_stepsize << endl;
        }
        else {  // good improvement
            change = true;
            X_current = X_updated;
            cout << "-- final stepsize " << stepsize << endl;
            cout << "-- Gauss-Newton old energy   " << energy_start_of_step << endl;
            cout << "-- Gauss-Newton final energy " << energy_updated << endl;
        }
    }

    // double check if new edge lenghts are ok
    // -------------------------------------
    // TODO outcomment everything below
    if (change) {
        X_restshape = X_current;
        double diff = 0;
        double diff_constraint = 0;
        for (int e = 0; e < edges; e++) {
            Vector3d dx = (X_restshape.row(E(e, 0)) - X_restshape.row(E(e, 1)));
            double l_now = dx.dot(dx);
            double single_diff = (l_target[e] * l_target[e] - l_now) * (l_target[e] * l_target[e] - l_now);
            diff += single_diff;
            //if (edge_constraints[e] > 0)
            //    diff_constraint += single_diff;
        }
        cout << "-- Final difference to wanted lenghts: " << diff << " difference before: " << original_diff << endl;
        //cout << "-- Final difference to wanted constrs: " << diff_constraint << " difference before: " << original_diff << endl;
    }
}




