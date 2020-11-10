#include "readGarment.h"

#include <string>
#include <iostream>
#include <algorithm>
#include <stb_image.h>
#include <experimental/filesystem>

#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/png/readPNG.h>

using namespace std;
using namespace Eigen;

string removeFileFromPath(string path) {
	size_t lastindex1 = path.find_last_of("/");  // account for both windows and ubuntu
	size_t lastindex2 = path.find_last_of("\\");
	size_t lastindex = min(lastindex1, lastindex2);
	return path.substr(0, lastindex + 1);
}

string getFileExtension(string file) {
	size_t lastindex = file.find_last_of(".");
	return file.substr(lastindex+1, file.length() );
}

string getFileName(string file) {
	size_t path_lastindex1 = file.find_last_of("/");  // account for both windows and ubuntu
	size_t path_lastindex2 = file.find_last_of("\\");
	size_t startindex = min(path_lastindex1, path_lastindex2) + 1;
	size_t lastindex = file.find_last_of(".");

	size_t collision_index = file.find_last_of("_");	// check if "_col" is included
	string col = file.substr(collision_index, 4);
	if (col.compare("_col") == 0)
		lastindex -= 4;

	return file.substr(startindex, lastindex-startindex);
}

// this function is from: https://github.com/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples
// Simple helper function to load an image into a OpenGL texture with common settings
bool LoadTextureFromFile(const char* filename, GLuint & out_texture, int & out_width, int & out_height)
{
	// Load from file
	int image_width = 0;
	int image_height = 0;
	unsigned char* image_data = stbi_load(filename, &image_width, &image_height, NULL, 4);
	if (image_data == NULL)
		return false;

	// Create a OpenGL texture identifier
	GLuint image_texture;
	glGenTextures(1, &image_texture);
	glBindTexture(GL_TEXTURE_2D, image_texture);

	// Setup filtering parameters for display
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Upload pixels into texture
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
	stbi_image_free(image_data);

	out_texture = image_texture;
	out_width = image_width;
	out_height = image_height;

	return true;
}

struct avatar {
	string name;
	MatrixXd V;
	MatrixXi F;
	bool mesh_set = false;
	GLuint I;
	bool img_set = false;
};

bool readMannequinsFromFolder(
	string path,
	vector<MatrixXd>& V,
	vector<MatrixXi>& F,
	vector<GLuint>& I
) {
	path = removeFileFromPath(path);
	vector< avatar > avatars; 

	V.clear();
	F.clear();
	I.clear();

	// load all files
	for (const auto& entry : std::experimental::filesystem::directory_iterator(path)) {
		// check the name of the file and see if we already loaded other correspondinf files
		string file = entry.path().string();
		string name = getFileName(file);

		int current_index = avatars.size();
		for (int a = 0; a < avatars.size(); a++) 
			if (avatars[a].name.compare(name) == 0)
				current_index = a;
		if (current_index == avatars.size()) {
			avatar av;
			av.name = name;
			avatars.push_back(av);
		}

		if (getFileExtension(file).compare("ply") == 0 || getFileExtension(file).compare("obj") == 0) {
			MatrixXd oneV;
			MatrixXi oneF;
			cout << "reading " << file << endl;
			if(getFileExtension(file).compare("ply") == 0) 
				igl::readPLY(file, oneV, oneF);
			else 
				igl::readOBJ(file, oneV, oneF);

			avatars[current_index].V = oneV;
			avatars[current_index].F = oneF;
			avatars[current_index].mesh_set = true;
			
		} else if (getFileExtension(file).compare("png") == 0) {
			cout << "reading " << file << endl;
			const char* char_file = file.c_str();
			GLuint texture;
			int w,h;
			LoadTextureFromFile(char_file, texture, w, h);
			avatars[current_index].I = texture;
			avatars[current_index].img_set = true;
		}
	}

	// check if everything was loaded correctly
	// delete entries that miss a mesh or an image
	// create collision meshes for all meshes which do not have one
	for (int a = 0; a < avatars.size(); a++) {
		if (!(avatars[a].mesh_set && avatars[a].img_set))
			continue;
	
		V.push_back(avatars[a].V);
		F.push_back(avatars[a].F);
		I.push_back(avatars[a].I);
	}

	return true;
}
