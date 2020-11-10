#include "dijkstra.h"

using namespace std;
using namespace Eigen;

void dijkstra(const int& vertex, const VectorXd& previous, vector<int>& path){
    int source = vertex;
    path.clear();
    for (; source != -1; source = previous[source])
        path.push_back(source);
}

int dijkstra(
        const MatrixXd & V,
        const vector< vector<int> > & A,
        const int & source,
        const int & target,
        VectorXd & min_distance,
        VectorXd & previous
) {
    int numV = A.size();

    min_distance.setConstant(numV, 1, numeric_limits<double>::infinity());
    min_distance[source] = 0;
    previous.setConstant(numV, 1, -1);
    set< pair<double, int> > vertex_queue;
    vertex_queue.emplace(min_distance[source], source);

    while(!vertex_queue.empty()) {
        double dist = vertex_queue.begin()->first;
        int u = vertex_queue.begin()->second;
        vertex_queue.erase(vertex_queue.begin());

        if (target == u)
            return u;

        // Visit each edge exiting u
        const vector<int>& neighbors = A[u];
        for (auto v : neighbors) {
            double distance_through_u = dist + (V.row(u) - V.row(v)).norm();
            if (distance_through_u < min_distance[v]) {
                vertex_queue.erase(make_pair(min_distance[v], v));
                min_distance[v] = distance_through_u;
                previous[v] = u;
                vertex_queue.emplace(min_distance[v], v);
            }
        }
    }
    return -1;
}
