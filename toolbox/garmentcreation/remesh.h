#ifndef remesh_h
#define remesh_h

#include <Eigen/Dense>

void remesh(const Eigen::MatrixXd& Vin, const Eigen::MatrixXi& Fin,
            Eigen::MatrixXd& Vout, Eigen::MatrixXi& Fout,
            const double refinementFactor = 2.);

#endif 
