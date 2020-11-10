#ifndef GARMENTSHAPE_READGARMENT_H
#define GARMENTSHAPE_READGARMENT_H

#include <vector>
#include <Eigen/Dense>

#ifdef _WIN32
	#include <windows.h>
	#include <GL/gl.h>
#elif __MACH__
	#include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
#endif

// read all files from a folder
// for each pose the following are required and need to have the same name:
//		one mesh in OBJ or PLY format												e.g. avatar_pose01.ply
//		one image in PNG format														e.g. avatar_pose01.png
bool readMannequinsFromFolder(
	const std::string path,
	std::vector<Eigen::MatrixXd>& V,
	std::vector<Eigen::MatrixXi>& F,
	std::vector<GLuint>& I);

#endif
