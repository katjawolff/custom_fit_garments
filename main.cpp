#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>

#include <iostream>
#include <chrono>

#include <igl/png/readPNG.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/edges.h>
#include <igl/boundary_loop.h>
#include <igl/bfs.h>
#include <igl/remove_duplicate_vertices.h>

#include "toolbox/readGarment.h"
#include "toolbox/clothsimulation/cloth.h"
#include "toolbox/mesh_interpolation.h"
#include "toolbox/garmentcreation/garment_boundaries.h"
#include "toolbox/adjacency.h"
#include "toolbox/garmentcreation/remesh.h"

using namespace Eigen;
using namespace std;

// variables
//float time_step = 0.0025;
float time_step = 0.0025;
float K_stretch = 800.f;            // in g/s^2
float K_stretch_damping = 100.;
float K_shear = 200.;
float K_shear_damping = 1.;
float K_bend = 0.001;
float K_bend_damping = 0.01;
float total_mass = 1.5;                // in kg;
float garment_refinement = 0.3;
float t_stretch = 1.09;


float t_compress = 0.;
float dist_to_body = 0.005;        

float rs_offset = 0.;

bool gar_loaded = false;
bool man_loaded = false;
bool cloth_loaded = false;
bool show_avatar_window = false;
bool man_of_many_loaded = false;
bool move_avatar = false;
bool adjust_garment = true;
bool show_wiregrid = false;
bool simulate = false;
bool pause_simulation = false;
bool paint = false;
bool gravity = true;

Cloth* clo;
Eigen::MatrixXd Vg, Vg_initial, Vm, Vuv;   // meshes for the garment and mannequin
Eigen::MatrixXi Fg, Fm;
Eigen::MatrixXi Eg;
vector< vector<int> > Am, Ag_ve;            // adjacency list for the mannequin mesh Vm/Fm // vertex-edge adjacency for the garment
SparseMatrix<double> Ag_mat;                // adjacency matrix for the garment
vector<MatrixXd> V_avatars;
vector<MatrixXi> F_avatars;
vector<GLuint> Images_avatars;
Eigen::Vector3d ambient, ambient_grey, diffuse, diffuse_grey, specular;

// avatar interpolation
MeshInterpolator* mesh_interpolator;
const int avatar_movement_total_frames = 60;
int avatar_movement_current_frame = 0;
int intermediate_frames_count = 0;
int max_sim_interval = 4; // frames

// garment paint more cloth
float paint_inner_radius = 0.01;
float paint_outer_radius = 0.03;
float max_edge_adjustment = 3.;
vector<double> edge_adjustment;
bool edge_adjustment_needed = false;
MatrixXd Cpaint;

// garment add new boundary
bool place_boundary = false;
Vector3d center, axis;
vector<int> old_boundary;
double avrg_dist_to_center;

// boundaries
GarmentBoundaries* boundaries;
GarmentBoundaries* cut_boundaries;

// mouse interaction
enum MouseMode { SELECTVERTS, SELECTBVERTS1, SELECTBVERTS2, SELECTGARMENT, SELECTBOUNDARY, PAINT, NEWBOUNDARY, NONE };
MouseMode mouse_mode = NONE;

// some extra functions we need, that are defined after the main function
bool computePointOnMesh(igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd& Vuse, Eigen::MatrixXi& Fuse,
    Eigen::Vector3d& position, int& fid);
int computeClosestVertexOnMesh(Vector3d& b, int& fid, MatrixXi& F);
bool callback_mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
bool callback_mouse_move(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
bool callback_key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifiers);

void avatarMenu(igl::opengl::glfw::Viewer& viewer);
void setNewGarmentMesh(igl::opengl::glfw::Viewer& viewer);
void setNewInterpolationMesh(MatrixXd& Vm_new);
bool interpolateAvatar(igl::opengl::glfw::Viewer& viewer);
void startSimulation(igl::opengl::glfw::Viewer& viewer);
void pauseSimulation(igl::opengl::glfw::Viewer& viewer);
void stopSimulation(igl::opengl::glfw::Viewer& viewer);

void showGarment(igl::opengl::glfw::Viewer& viewer);
void showMannequin(igl::opengl::glfw::Viewer& viewer);
void showBoundary(igl::opengl::glfw::Viewer& viewer);
void showRestShape(igl::opengl::glfw::Viewer& viewer);


int left_view, right_view;

bool pre_draw(igl::opengl::glfw::Viewer& viewer) {

    viewer.data(0).set_visible(true, left_view);
    viewer.data(0).set_visible(false, right_view);

    viewer.data(1).set_visible(true, left_view);
    viewer.data(1).set_visible(false, right_view);

    viewer.data(2).set_visible(true, left_view);
    viewer.data(2).set_visible(false, right_view);

    viewer.data(3).set_visible(false, left_view);
    viewer.data(3).set_visible(true, right_view);

    // cloth simulation
    if (simulate) {
        // update the avatar
        bool avatar_moved = false;
        if (intermediate_frames_count < max_sim_interval)
            intermediate_frames_count++;
        else {
            intermediate_frames_count = 0;
            if (move_avatar) avatar_moved = interpolateAvatar(viewer);
        }

        if (avatar_moved)
            clo->StepPhysics(time_step, Vm, Fm);    // if the avatar moved, we need to update the collision mesh
        else
            clo->StepPhysics(time_step);    // normal, faster update
        if (adjust_garment && intermediate_frames_count == max_sim_interval - 1) {
            clo->updateRestShape(Vg, t_stretch, t_compress, edge_adjustment);

            fill(edge_adjustment.begin(), edge_adjustment.end(), 1.);   // reset paint
            for (int j = 0; j < Vg.rows(); j++)
                Cpaint.row(j) = RowVector3d(1, 1, 0);
        }
        MatrixXd U = clo->getMesh();

        // set stretch colors
        VectorXd S = clo->ComputeStretchPerTriangle();  // S(f) should be one -> bigger: stretch, smaller: compression
        MatrixXd C(Fg.rows(), 3);
        for (int f = 0; f < Fg.rows(); f++) {
            double y = (S(f) - 1.) * 3.;
            C.row(f) = Vector3d(1.0 + y, 1. - y, 0.0);
            //double y = S(f);
            //C.row(f) = Vector3d(y, 1.0 - y, 0.0);
        }

        viewer.selected_data_index = 0;
        viewer.data().set_vertices(U);
        viewer.data().compute_normals();
        viewer.data().set_colors(C);
        viewer.data().uniform_colors(ambient, diffuse, specular);
        viewer.data().set_face_based(false);
        viewer.data().show_texture = false;

        viewer.selected_data_index = 3;
        MatrixXd Vrs = Vg;
        Vrs.col(0).array() += rs_offset;
        viewer.data().set_vertices(Vrs);
        viewer.data().compute_normals();
        viewer.data().uniform_colors(ambient, diffuse, specular);
        viewer.data().show_texture = false;
        viewer.data().set_face_based(false);
    }

    // show the window with all loaded avatars
    if (show_avatar_window) avatarMenu(viewer);

    return false;
}


int main(int argc, char* argv[]) {

    // viewer
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    viewer.callback_init = [&](igl::opengl::glfw::Viewer& viewer)
    {
        viewer.core().viewport = Eigen::Vector4f(0, 0, 640, 800);
        left_view = viewer.core_list[0].id;
        right_view = viewer.append_core(Eigen::Vector4f(640, 0, 640, 800));
        return false;
    };

    viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer& v, int w, int h) {
        v.core(left_view).viewport = Eigen::Vector4f(0, 0, w / 2, h);
        v.core(right_view).viewport = Eigen::Vector4f(w / 2, 0, w - (w / 2), h);
        return true;
    };

    viewer.append_mesh();   // mesh for the garment
    viewer.append_mesh();   // mesh for the mannequin
    viewer.append_mesh();   // mesh for edge visualization (cylinders around edges)
    viewer.append_mesh();   // mesh for the garment rest shape

    // Layout
    viewer.core().background_color = Eigen::Vector4f(1, 1, 1, 1);
    viewer.core().rotation_type = igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_TRACKBALL;
    viewer.data().show_lines = false;
    ambient_grey = Vector3d(0.4, 0.4, 0.4);
    ambient = Vector3d(0.26, 0.26, 0.26);
    diffuse_grey = Vector3d(0.5, 0.5, 0.5);
    diffuse = Vector3d(0.4, 0.57, 0.66);    // blue
    specular = Vector3d(0.01, 0.01, 0.01);

    // Add content to the default menu window
    viewer.callback_pre_draw = &pre_draw;
    viewer.core().animation_max_fps = 30;
    viewer.core().is_animating = false;
    menu.callback_draw_viewer_menu = [&]() {
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Cloth", ImGuiTreeNodeFlags_DefaultOpen)) {

            float gui_w = ImGui::GetContentRegionAvailWidth();
            float gui_p = ImGui::GetStyle().FramePadding.x;

            if (ImGui::Button("Load Garment OBJ", ImVec2(gui_w - gui_p, 0))) {
                string garment_file_name = igl::file_dialog_open();

                igl::readOBJ(garment_file_name, Vg, Fg);

                if (Vg.rows() == 0 || Fg.rows() == 0) {
                    fprintf(stderr, "IOError: Could not load garment...\n");
                    return;
                }
                setNewGarmentMesh(viewer);
            }
            if (ImGui::Button("Load Mannequin OBJ", ImVec2(gui_w - gui_p, 0))) {
                cout << "Choose the file used for rendering the mannequin..." << endl;
                string mannequin_file_name = igl::file_dialog_open();

                igl::readOBJ(mannequin_file_name, Vm, Fm);

                if (Vm.rows() > 0) {
                    showMannequin(viewer);
                    boundaries = new GarmentBoundaries(Fm);
                    man_loaded = true;
                    man_of_many_loaded = false;
                }
                if (gar_loaded) setNewGarmentMesh(viewer);
            }
            if (ImGui::Button("Load Avatars from folder", ImVec2(gui_w - gui_p, 0))) {
                cout << "Choose one file from the folder you want to load..." << endl;
                string folder_path = igl::file_dialog_open();

                readMannequinsFromFolder(folder_path, V_avatars, F_avatars, Images_avatars);

                viewer.core().is_animating = true;
                show_avatar_window = true;
                man_of_many_loaded = false;
            }
            if (ImGui::Button("Save Restshape", ImVec2(gui_w - gui_p, 0))) {
                if (cloth_loaded) {
                    vector<MatrixXd> Vg_list;
                    vector<MatrixXi> Fg_list;
                    if (cut_boundaries->numberOfBoundaries() > 0)
                        cut_boundaries->cutGarmentAlongBoundaries(Vg, Vg_list, Fg_list);
                    else {
                        Vg_list.push_back(Vg);
                        Fg_list.push_back(Fg);
                    }

                    for (int i = 0; i < Vg_list.size(); i++) {
                        string folder_path = igl::file_dialog_save();
                        igl::writeOBJ(folder_path, Vg_list[i], Fg_list[i]);
                    }
                }
            }
            if (ImGui::Button("Save Garment", ImVec2(gui_w - gui_p, 0))) {
                if (cloth_loaded) {
                    MatrixXd U = clo->getMesh();
                    string folder_path = igl::file_dialog_save();
                    igl::writeOBJ(folder_path, U, Fg);

                    VectorXd S = clo->ComputeStretchPerTriangle();  // S(f) should be one -> bigger: stretch, smaller: compression
                    MatrixXd C(U.rows(), 3);
                    vector< vector<int> > vf_adj;
                    createVertexFaceAdjacencyList(Fg, vf_adj);
                    for (int v = 0; v < U.rows(); v++) {
                        C.row(v) = Vector3d::Zero();
                        for (int f = 0; f < vf_adj[v].size(); f++) {
                            double y = (S(vf_adj[v][f]) - 1.) * 3.;
                            double r = max(min(1.0 + y, 1.), 0.);
                            double g = max(min(1. - y, 1.), 0.);
                            C.row(v) += Vector3d(r, g, 0.0) * 255.;
                        }
                        C.row(v) /= vf_adj[v].size();
                    }
                    igl::writeOFF(folder_path, U, Fg, C);
                }
            }
            if (ImGui::Button("Save Mannequin", ImVec2(gui_w - gui_p, 0))) {
                string folder_path = igl::file_dialog_save();
                igl::writeOBJ(folder_path, Vm, Fm);
            }

            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
            ImGui::DragFloat("mass", &(total_mass), 0.f, 0.0f, 10.0f);
            ImGui::DragFloat("stretch", &(K_stretch), 1000.f, 0.0f, 1000000.0f);
            ImGui::DragFloat("st. damp.", &(K_stretch_damping), 10.f, 0.0f, 1000000.0f);
            ImGui::DragFloat("shear", &(K_shear), 0.f, 0.0f, 1000000.0f);
            ImGui::DragFloat("sh. damp.", &(K_shear_damping), 0.f, 0.0f, 1000000.0f);
            ImGui::DragFloat("bend /1K", &(K_bend), 0.f, 0.0f, 1000.0f);
            ImGui::DragFloat("bd. damp /1K", &(K_bend_damping), 0.0f, 0.0f, 1000.0f);
            ImGui::DragFloat("offset from body", &(dist_to_body), 0.f, 0.0f, 1.0f);
            ImGui::DragFloat("time_step", &(time_step), 0.f, 0.0f, 1.0f);
            ImGui::DragInt("movement sim steps", &(max_sim_interval), 0.f, 0.0f, 1.0f);
            ImGui::PopItemWidth();

            ImGui::Checkbox("Adjust garment", &adjust_garment);
            //ImGui::Checkbox("Gravity", &gravity);
            ImGui::DragFloat("Threshhold strech", &(t_stretch), 0.f, 0.0f, 2.0f);

            if (ImGui::Button("Start", ImVec2(gui_w - gui_p, 0))) {
                startSimulation(viewer);
            }
            if (ImGui::Button("Pause", ImVec2(gui_w - gui_p, 0))) {
                pauseSimulation(viewer);
            }
            if (ImGui::Button("Stop", ImVec2(gui_w - gui_p, 0))) {
                stopSimulation(viewer);
            }
            if (ImGui::Button("Adjust offset", ImVec2(gui_w - gui_p, 0))) {
                clo->setOffset(dist_to_body);
            }
            if (ImGui::Button("Load Sim OBJ", ImVec2(gui_w - gui_p, 0))) {
                string garment_file_name = igl::file_dialog_open();

                MatrixXd Vsim;
                MatrixXi Fsim;
                igl::readOBJ(garment_file_name, Vsim, Fsim);

                if (Vsim.rows() == 0 || Fsim.rows() == 0) {
                    fprintf(stderr, "IOError: Could not load garment...\n");
                    return;
                }
                clo->setSimMesh(Vsim);

                // check if there are several components. If needed, add seam forces
                vector<vector<int>> L;
                igl::boundary_loop(Fsim, L);
                if (L.size() > 1) {
                    vector< pair<int, int> > seam_verts;

                    for (int b1 = 0; b1 < L.size() - 1; b1++) {         // TODO do this better with a kd tree or sth similar
                        for (int v1 = 0; v1 < L[b1].size(); v1++) {
                            bool found = false;
                            for (int b2 = b1 + 1; b2 < L.size() && !found; b2++) {
                                for (int v2 = 0; v2 < L[b2].size() && !found; v2++) {
                                    if ((Vsim.row(L[b1][v1]) - Vsim.row(L[b2][v2])).norm() == 0 /*< 1e-8*/) {
                                        seam_verts.push_back(make_pair(L[b1][v1], L[b2][v2]));
                                        found = true;
                                    }
                                }
                            }
                        }
                    }

                    clo->setSeamVertices(seam_verts);
                }
            }
        }

        if (ImGui::CollapsingHeader("Boundaries", ImGuiTreeNodeFlags_DefaultOpen)) {
            float gui_w = ImGui::GetContentRegionAvailWidth();
            float gui_p = ImGui::GetStyle().FramePadding.x;

            if (ImGui::Button("Start new circle boundary", ImVec2(gui_w - gui_p, 0))) {
                mouse_mode = SELECTVERTS;
            }
            if (ImGui::Button("Close circle boundary", ImVec2(gui_w - gui_p, 0))) {
                if (!gar_loaded)
                    boundaries->closeBoundary(Vm);
                else {
                    cut_boundaries->closeBoundary(Vg);
                }
                showBoundary(viewer);
                mouse_mode = NONE;
            }
            /*if (ImGui::Button("Start new edge boundary", ImVec2(gui_w - gui_p, 0))) {
                mouse_mode = SELECTBVERTS1;
            }*/
            if (ImGui::Button("Delete last", ImVec2(gui_w - gui_p, 0))) {
                if (!gar_loaded) boundaries->deleteLast();
                else cut_boundaries->deleteLast();
                showBoundary(viewer);
            }
            if (ImGui::Button("Delete all", ImVec2(gui_w - gui_p, 0))) {
                if (!gar_loaded) boundaries->deleteAll();
                else cut_boundaries->deleteAll();
                showBoundary(viewer);
            }
            if (ImGui::Button("Save boundaries", ImVec2(gui_w - gui_p, 0))) {
                string folder_path = igl::file_dialog_save();
                boundaries->saveBoundaries(folder_path);
            }
            if (ImGui::Button("Load boundaries", ImVec2(gui_w - gui_p, 0))) {
                string folder_path = igl::file_dialog_open();
                boundaries->loadBoundaries(folder_path);
                showBoundary(viewer);
            }
            if (ImGui::Button("Duplicate boundary", ImVec2(gui_w - gui_p, 0))) {
                mouse_mode = NEWBOUNDARY;
            }
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
            ImGui::DragFloat("garment refinement", &(garment_refinement), 0.f, 0.0f, 5.0f);
            ImGui::PopItemWidth();
            if (ImGui::Button("Garment from boundaries", ImVec2(gui_w - gui_p, 0))) {
                viewer.core().is_animating = false;
                cloth_loaded = false;
                mouse_mode = SELECTGARMENT;
                cout << "Click on the body part that needs clothing." << endl;
            }
            if (ImGui::Button("Fix garment at boundary", ImVec2(gui_w - gui_p, 0))) {
                mouse_mode = SELECTBOUNDARY;
                cout << "Click on the boundary that you want to fix." << endl;
            }

            if (ImGui::Button("PAINT", ImVec2(gui_w - gui_p, 0))) {
                mouse_mode = PAINT;
                cout << "Paint now." << endl;
            }

        }};

    // add mouse handling
    viewer.callback_mouse_down = callback_mouse_down;
    viewer.callback_mouse_move = callback_mouse_move;
    viewer.callback_mouse_up = callback_mouse_up;
    viewer.callback_key_down = callback_key_down;

    // Plot the mesh
    viewer.launch();
}

void avatarMenu(igl::opengl::glfw::Viewer& viewer) {

    bool* p_open = new bool;    // ;_;
    *p_open = true;

    ImGui::SetNextWindowSize(ImVec2(470, 500), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Avatars", p_open)) {
        ImGui::End();
        return;
    }

    float gui_w = ImGui::GetContentRegionAvailWidth();
    float gui_p = ImGui::GetStyle().FramePadding.x;

    int images = Images_avatars.size();

    for (int i = 0; i < images; i++) {
        GLuint tex = Images_avatars[i];

        //if(i%2 == 1)
          //  ImGui::SameLine();

        if (ImGui::ImageButton((void*)(intptr_t)tex, ImVec2(200, 200), ImVec2(0, 0), ImVec2(1.f, 1.f), gui_p, ImColor(0, 0, 0, 255))) {

            // if we didn't load any of these mannequins yet, just show this one
            // also if we don't animate a piece of cloth, just switch to the next pose
            if (!man_of_many_loaded || !simulate) {
                Vm = V_avatars[i];
                Fm = F_avatars[i];

                showMannequin(viewer);
                if (!man_of_many_loaded)
                    boundaries = new GarmentBoundaries(Fm);
                else
                    showBoundary(viewer);
                man_loaded = true;
                man_of_many_loaded = true;
                if (gar_loaded) setNewGarmentMesh(viewer);
            }
            // but if we already loaded one mannequin, interpolate to this one now
            else {
                setNewInterpolationMesh(V_avatars[i]);
            }
        }
    }

    ImGui::End();
    delete p_open;
}

void showGarment(igl::opengl::glfw::Viewer& viewer) {
    viewer.selected_data_index = 0;
    viewer.data().clear();
    viewer.data().set_mesh(Vg, Fg);
    viewer.data().uniform_colors(ambient, diffuse, specular);
    viewer.data().show_texture = false;
    viewer.data().set_face_based(false);
}
void showMannequin(igl::opengl::glfw::Viewer& viewer) {
    viewer.selected_data_index = 1;
    viewer.data().clear();
    viewer.data().set_mesh(Vm, Fm);
    viewer.data().show_lines = false;
    viewer.data().uniform_colors(ambient_grey, diffuse_grey, specular);
    //viewer.core().align_camera_center(viewer.data().V, viewer.data().F);
    viewer.data().show_texture = false;
    viewer.data().set_face_based(false);
}
void showBoundary(igl::opengl::glfw::Viewer& viewer) {
    MatrixXd Vb;
    MatrixXi Fb;
    boundaries->getCylindersAroundBoundaryEdges(Vm, 0.002, Vb, Fb);
    //if (gar_loaded) {
    //    cut_boundaries->getCylindersAroundBoundaryEdges(Vg, 0.002, Vb, Fb); // only show cut boundaries for now if we have
    //    Vb.col(0).array() += rs_offset;
    //}

    viewer.selected_data_index = 2;
    viewer.data().clear();
    viewer.data().set_mesh(Vb, Fb);
    viewer.data().show_lines = false;
    viewer.data().show_texture = false;
    viewer.data().uniform_colors(ambient, Vector3d(1.0, 0.0, 0.0), specular);
}
void showRestShape(igl::opengl::glfw::Viewer& viewer) {
    viewer.selected_data_index = 3;
    viewer.data().clear();

    MatrixXd Vrs = Vg;
    Vrs.col(0).array() += rs_offset;

    viewer.data().set_mesh(Vrs, Fg);
    viewer.data().uniform_colors(ambient, diffuse, specular);
    viewer.data().show_texture = false;
    viewer.data().set_face_based(false);
}

void setNewGarmentMesh(igl::opengl::glfw::Viewer& viewer) {
    Vg_initial = Vg;
    igl::edges(Fg, Eg);
    showGarment(viewer);
    showRestShape(viewer);
    edge_adjustment.resize(Eg.rows(), 1.);
    Cpaint.resize(Vg.rows(), 3);
    for (int j = 0; j < Vg.rows(); j++)
        Cpaint.row(j) = RowVector3d(1, 1, 0);
    cut_boundaries = new GarmentBoundaries(Fg);
    gar_loaded = true;

    if (man_loaded) {
        showBoundary(viewer);
        double k_bnd = K_bend / 1000.;
        double k_bnd_dmp = K_bend_damping / 1000.;
        clo = new Cloth(Vg, Fg, Vm, Fm, total_mass, K_stretch, K_stretch_damping, K_shear, K_shear_damping, k_bnd,
            k_bnd_dmp, dist_to_body, man_loaded);
        clo->setGravity(gravity);
        boundaries->createCorrespondences(Vm, Vg, Fg);
        clo->setConstrainedVertices(boundaries);
        cloth_loaded = true;
    }
}
void setNewInterpolationMesh(MatrixXd& Vm_new) {
    mesh_interpolator = new MeshInterpolator(Vm, Vm_new, Fm);
    //mesh_interpolator->precomputeInterpolatedMeshes(avatar_movement_total_frames);
    avatar_movement_current_frame = avatar_movement_total_frames - 1;
    move_avatar = true;
}
void startSimulation(igl::opengl::glfw::Viewer& viewer) {
    intermediate_frames_count = 0;
    if (pause_simulation) pause_simulation = false;
    simulate = true;
    viewer.core().is_animating = true;
}
void pauseSimulation(igl::opengl::glfw::Viewer& viewer) {
    simulate = false;
    pause_simulation = true;
    move_avatar = false;
    viewer.core().is_animating = false;
}
void stopSimulation(igl::opengl::glfw::Viewer& viewer) {
    pause_simulation = false;
    simulate = false;
    cloth_loaded = false;
    viewer.core().is_animating = false;

    Vg = Vg_initial;
    setNewGarmentMesh(viewer);
    showGarment(viewer);
}

bool interpolateAvatar(igl::opengl::glfw::Viewer& viewer) {
    // get the current interpolation value p
    // TODO do this with times instead of frames later, when everything runs smoothly
    double p = float(avatar_movement_total_frames - avatar_movement_current_frame) / float(avatar_movement_total_frames);
    mesh_interpolator->interpolatedMesh(p, Vm);
    //mesh_interpolator->getPrecomputedMesh(avatar_movement_current_frame, Vm);

    showMannequin(viewer);
    showBoundary(viewer);   // updates the boundary vizualization on the moved avatar

    // update the time value
    if (avatar_movement_current_frame == 0)
        move_avatar = false;
    else
        avatar_movement_current_frame -= 1;

    return true;
}
void getBoundaryCenterAxis(Vector3d& v) {
    vector<vector<int> > boundaries;
    igl::boundary_loop(Fg, boundaries);

    // get coordinates of the clostest boundaries
    double dist = 1e10;
    double boundary_id = 0;

    for (int i = 0; i < boundaries.size(); i++) {
        for (int j = 0; j < boundaries[i].size(); j++) {
            double new_dist = (Vg.row(boundaries[i][j]).transpose() - v).norm();
            if (new_dist < dist) {
                dist = new_dist;
                boundary_id = i;
            }
        }
    }
    old_boundary = boundaries[boundary_id];

    // go along boundary and average vertices and normals
    center = Vector3d(0, 0, 0);
    for (int j = 0; j < old_boundary.size(); j++)
        center += Vg.row(old_boundary[j]);
    center /= old_boundary.size();

    axis = Vector3d(0, 0, 0);
    avrg_dist_to_center = 0;
    for (int j = 0; j < old_boundary.size(); j++) {
        Vector3d to_center = center - Vg.row(old_boundary[j]).transpose();
        avrg_dist_to_center += to_center.norm();
        Vector3d edge;
        if (j == 0) edge = Vg.row(old_boundary[j]) - Vg.row(old_boundary[old_boundary.size() - 1]);
        else edge = Vg.row(old_boundary[j]) - Vg.row(old_boundary[j - 1]);
        axis += to_center.cross(edge).normalized();
    }
    avrg_dist_to_center /= old_boundary.size();
    axis.normalize();
}
void duplicateBoundaryAndCreateMesh(const double scaling, const Vector3d& new_center, MatrixXd& Vnew, MatrixXi& Fnew) {
    // duplicate
    MatrixXd new_boundary(old_boundary.size(), 3);
    for (int j = 0; j < old_boundary.size(); j++) {
        Vector3d to_center = Vg.row(old_boundary[j]).transpose() - center;
        to_center *= scaling;
        new_boundary.row(j) = new_center + to_center;
    }

    // create triangles between old and new boundary
    Vnew.resize(old_boundary.size() * 2, 3);
    Fnew.resize(old_boundary.size() * 2, 3);
    for (int i = 0; i < old_boundary.size(); i++) {
        Vnew.row(2 * i) = Vg.row(old_boundary[i]);
        Vnew.row(2 * i + 1) = new_boundary.row(i);
        Fnew.row(2 * i) = Vector3i(2 * i, 2 * i + 1, 2 * i + 2);
        Fnew.row(2 * i + 1) = Vector3i(2 * i + 1, 2 * i + 3, 2 * i + 2);
        if (i == old_boundary.size() - 1) {
            Fnew.row(2 * i) = Vector3i(2 * i, 2 * i + 1, 0);
            Fnew.row(2 * i + 1) = Vector3i(2 * i + 1, 1, 0);
        }
    }
}

bool computePointOnMesh(igl::opengl::glfw::Viewer& viewer, MatrixXd& V, MatrixXi& F, Vector3d& b, int& fid) {
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    return igl::unproject_onto_mesh(Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, fid, b);
}
int computeClosestVertexOnMesh(Vector3d& b, int& fid, MatrixXi& F) {
    // get the closest vertex in that face
    int v_id;
    if (b(0) > b(1) && b(0) > b(2))
        v_id = F(fid, 0);
    else if (b(1) > b(0) && b(1) > b(2))
        v_id = F(fid, 1);
    else
        v_id = F(fid, 2);
    return v_id;
}

bool callback_mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
    if (button == (int)igl::opengl::glfw::Viewer::MouseButton::Right)
        return false;

    // select vertices for the boundary
    if (mouse_mode == SELECTVERTS)
    {
        int fid;
        Eigen::Vector3d b;
        // Boundaries on avatar
        if (!gar_loaded) {
            if (computePointOnMesh(viewer, Vm, Fm, b, fid)) {
                int v_id = computeClosestVertexOnMesh(b, fid, Fm);
                boundaries->addPointsToBoundary(Vm, v_id);
                showBoundary(viewer);
                viewer.data().set_points(Vm.row(v_id), RowVector3d(1.0, 0.0, 0.0));
                return true;
            }
        }
        // Boundaries on garment
        else {
            MatrixXd Vrs = Vg;
            Vrs.col(0).array() += rs_offset;

            if (computePointOnMesh(viewer, Vrs, Fg, b, fid)) {
                int v_id = computeClosestVertexOnMesh(b, fid, Fg);
                cut_boundaries->addPointsToBoundary(Vrs, v_id);
                showBoundary(viewer);
                viewer.data().set_points(Vrs.row(v_id), RowVector3d(1.0, 0.0, 0.0));
                return true;
            }
        }
    }
    // select an edge to edge boundary
    // unused at the moment
    if (mouse_mode == SELECTBVERTS1) {
        int fid;
        Eigen::Vector3d b;
        if (computePointOnMesh(viewer, Vg, Fg, b, fid)) {
            int v_id = computeClosestVertexOnMesh(b, fid, Fg);
            // TODO find closest boundary vertex OR just check if it is one
            cut_boundaries->addPointsToBoundary(Vg, v_id);
            viewer.data().set_points(Vg.row(v_id), RowVector3d(1.0, 0.0, 0.0));
            mouse_mode = SELECTBVERTS2;
            return true;
        }
    }
    if (mouse_mode == SELECTBVERTS2) {
        int fid;
        Eigen::Vector3d b;
        if (computePointOnMesh(viewer, Vg, Fg, b, fid)) {
            int v_id = computeClosestVertexOnMesh(b, fid, Fg);
            // TODO find closest boundary vertex OR just check if it is one
            cut_boundaries->addPointsToBoundary(Vg, v_id);
            cut_boundaries->endBoundary(Vg);
            showBoundary(viewer);
            mouse_mode = SELECTBVERTS2;
            return true;
        }
    }
    // select a region that becomes be a garment
    if (mouse_mode == SELECTGARMENT) {
        int fid;
        Eigen::Vector3d b;
        if (computePointOnMesh(viewer, Vm, Fm, b, fid)) {
            int v_id = computeClosestVertexOnMesh(b, fid, Fm);
            boundaries->garmentFromBoundaries(Vm, v_id, Vg, Fg, garment_refinement);
            setNewGarmentMesh(viewer);
            mouse_mode = NONE;
            return true;
        }
    }
    // mark a boundary as fixed
    if (mouse_mode == SELECTBOUNDARY) {
        int fid;
        Vector3d b;
        if (computePointOnMesh(viewer, Vm, Fm, b, fid)) {
            Vector3d p = b(0) * Vm.row(Fm(fid, 0)) + b(1) * Vm.row(Fm(fid, 1)) + b(2) * Vm.row(Fm(fid, 2));
            boundaries->markClosestBoundaryAsFixed(Vm, p);
            // TODO show boundary as blue
            mouse_mode = NONE;
            return true;
        }
    }
    // paint more cloth
    if (mouse_mode == PAINT) {
        cout << "paintstart" << endl;
        paint = true;
        igl::adjacency_matrix(Fg, Ag_mat);
        createVertexEdgeAdjecencyList(Eg, Ag_ve);
        mouse_mode = NONE;
    }
    // select a boundary of the garment that should be duplicated ...
    if (mouse_mode == NEWBOUNDARY) {
        cout << "click on the garment near a bounary that needs extending" << endl;
        int fid;
        Eigen::Vector3d b;
        if (computePointOnMesh(viewer, Vm, Fm, b, fid)) {
            Vector3d p = b(0) * Vm.row(Fm(fid, 0)) + b(1) * Vm.row(Fm(fid, 1)) + b(2) * Vm.row(Fm(fid, 2));
            getBoundaryCenterAxis(p);
            place_boundary = true;
            mouse_mode = NONE;
        }
        return true;
    }
    // ...and duplicate the boundary to create a bigger garment
    if (place_boundary) {
        cout << "create new boundary" << endl;

        // get click axis
        Vector3d s, dir;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        igl::unproject_ray(Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, s, dir);

        // get distance to boundary axis
        Vector3d n = dir.cross(axis);
        double distance = n.dot(center - s) / n.norm();
        Vector3d n1 = dir.cross(n);
        Vector3d new_center = center + (s - center).dot(n1) * axis / axis.dot(n1);
        double scaling = abs(distance / avrg_dist_to_center);

        // add new boundary
        MatrixXd Vnew;
        MatrixXi Fnew;
        duplicateBoundaryAndCreateMesh(scaling, new_center, Vnew, Fnew);

        // add to Vm
        MatrixXd Vmerged(Vg.rows() + Vnew.rows(), 3);
        MatrixXi Fmerged(Fg.rows() + Fnew.rows(), 3);
        Vmerged.block(0, 0, Vg.rows(), 3) = Vg;
        Vmerged.block(Vg.rows(), 0, Vnew.rows(), 3) = Vnew;
        Fmerged.block(0, 0, Fg.rows(), 3) = Fg;
        Fmerged.block(Fg.rows(), 0, Vnew.rows(), 3) = Fnew;
        for (int f = Fg.rows(); f < Fmerged.rows(); f++)
            Fmerged.row(f) += RowVector3i(1, 1, 1) * Vg.rows();

        // remove duplicate vertices to make into single connected mesh
        MatrixXd SV;
        MatrixXi SF;
        VectorXi SVI, SVJ;
        igl::remove_duplicate_vertices(Vmerged, Fmerged, 1e-7, SV, SVI, SVJ, SF);

        // remesh
        double area_old = 1., area_new = 1.;
        for (int f = 0; f < Fg.rows(); f++) {
            Vector3d v1 = Vg.row(Fg(f, 1)) - Vg.row(Fg(f, 0));
            Vector3d v2 = Vg.row(Fg(f, 2)) - Vg.row(Fg(f, 0));
            area_old += v1.cross(v2).norm();
        }
        for (int f = 0; f < Fnew.rows(); f++) {
            Vector3d v1 = Vnew.row(Fnew(f, 1)) - Vnew.row(Fnew(f, 0));
            Vector3d v2 = Vnew.row(Fnew(f, 2)) - Vnew.row(Fnew(f, 0));
            area_old += v1.cross(v2).norm();
        }

        double remesh_factor = (area_new + area_old) / area_old;
        remesh(SV, SF, Vg, Fg, remesh_factor);

        setNewGarmentMesh(viewer);
        place_boundary = false;
        return true;
    }

    return false;
}
bool callback_mouse_move(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y)
{
    // paint the garment
    if (paint && gar_loaded) {
        int fid;
        Eigen::Vector3d b;

        MatrixXd U;
        if (cloth_loaded) U = clo->getMesh();
        else U = Vg;

        if (computePointOnMesh(viewer, U, Fg, b, fid)) {
            Vector3d point = b(0) * U.row(Fg(fid, 0)) + b(1) * U.row(Fg(fid, 1)) + b(2) * U.row(Fg(fid, 2));
            int v_id = computeClosestVertexOnMesh(b, fid, Fg);

            // do breadth first search to mark close edges with the paintbrush
            vector<int> d, predecessors;    // d: indices of vertices in discovery order, p: indices of predecessors (-1 is root),
                                            //p[i] is the index of the vertex v which preceded d[i] in the breadth first traversal
            igl::bfs(Ag_mat, v_id, d, predecessors);

            // go through edges
            int i = 0;
            vector<bool> edge_visited(Eg.rows(), false);
            double vertex_dist = 0.;
            while (vertex_dist < 2. * paint_outer_radius && i < d.size()) {
                int v = d[i];                           // get next vertex id
                vector<int> adjacent_edges = Ag_ve[v];  // get all adjacent edges (mark those that were already visited)
                for (int j = 0; j < adjacent_edges.size(); j++) {
                    int e = adjacent_edges[j];
                    if (!edge_visited[e]) {
                        // get distance from point (middle of edge)
                        int v1 = Eg(e, 0);
                        int v2 = Eg(e, 1);
                        Vector3d middle = 0.5 * (U.row(v1) + U.row(v2));
                        double dist = (middle - point).norm();
                        double value;
                        if (dist < paint_inner_radius) value = 1.;
                        else if (dist > paint_outer_radius) value = 0.;
                        else value = 1. - (dist - paint_inner_radius) / paint_outer_radius;
                        value *= max_edge_adjustment * 0.33; // don't use full saturation basically
                        /*edge_adjustment[e] = max(value, edge_adjustment[e]);
                        if (value > 0.) edge_adjustment_needed = true;*/
                        value += 1.; // make it a factor
                        edge_adjustment[e] = max(value, edge_adjustment[e]);
                        if (value > 0.) edge_adjustment_needed = true;

                        // color
                        Cpaint(v1, 1) = max(0., Cpaint(v1, 1) - (edge_adjustment[e] - 1.) / Ag_ve[v1].size());
                        Cpaint(v2, 1) = max(0., Cpaint(v2, 1) - (edge_adjustment[e] - 1.) / Ag_ve[v2].size());

                        // assign a value in (0,1) to this edge
                        edge_visited[e] = true;
                    }
                }
                // get distance of the vert is bigger than a threshold -> stop
                vertex_dist = (U.row(v).transpose() - point).norm();
                i++;
            }

            // render
            viewer.selected_data_index = 0;
            viewer.data().clear();
            viewer.data().set_mesh(U, Fg);
            viewer.data().uniform_colors(ambient, diffuse, specular);
            viewer.data().set_colors(Cpaint);
            viewer.data().show_texture = false;

            return true;
        }
    }

    return false;
}
bool callback_mouse_up(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
    if (paint) {
        paint = false;
        cout << "paint stop" << endl;
        return true;
    }
    return false;
};

bool callback_key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifiers) {
    bool handled = false;

    if (key == 'L') {
        show_wiregrid = !show_wiregrid;
        viewer.selected_data_index = 0;
        viewer.data().show_lines = show_wiregrid;
        viewer.selected_data_index = 3;
        viewer.data().show_lines = !show_wiregrid;

        handled = true;
    }

    if (key == 'S')
    {
        pauseSimulation(viewer);
        handled = true;
    }

    return handled;
}