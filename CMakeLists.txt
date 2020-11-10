cmake_minimum_required(VERSION 3.1)
project(garmentshape)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##############
### LIBIGL ###
##############
set(LIBIGL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ext/libigl/include")
list(APPEND CMAKE_MODULE_PATH "${LIBIGL_INCLUDE_DIR}/../cmake")

option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          ON)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

include(libigl)

###########
### PMP ###
###########
set(PMP_VERSION 1.2.0)

# setup STB Image
set(STBI_SOURCE_DIR "ext/pmp/external/stb_image")
include_directories(${STBI_SOURCE_DIR})
add_subdirectory(${STBI_SOURCE_DIR})

# setup PLY
set(RPLY_SOURCE_DIR "ext/pmp/external/rply")
include_directories(${RPLY_SOURCE_DIR})
add_subdirectory(${RPLY_SOURCE_DIR})

# setup Eigen
set(EIGEN_SOURCE_DIR "ext/pmp/external/eigen")
include_directories(${EIGEN_SOURCE_DIR})

include(AddFileDependencies)
include_directories(${PROJECT_SOURCE_DIR}/ext/pmp/src/)

# set default compiler flags
if((UNIX OR APPLE) AND CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic -Wall -Wextra")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7.0.0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context")
  endif()
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
endif()

if(APPLE)
	include_directories("/usr/local/include/")
endif()

# which directories to process
add_subdirectory(ext/pmp/src/pmp)

############################
### ADDITIONAL LIBRARIES ###
############################

# OpenMP
find_package(OpenMP REQUIRED)

### GARMENT SHAPE PROJECT ###
file(GLOB SRCFILES main.cpp
        toolbox/adjacency.cpp                               toolbox/adjacency.h
        toolbox/mesh_interpolation.cpp                      toolbox/mesh_interpolation.h
        toolbox/linear_solver.cpp                           toolbox/linear_solver.h
        toolbox/timer.cpp                                   toolbox/timer.h
        toolbox/readGarment.cpp                             toolbox/readGarment.h

        toolbox/clothsimulation/seam_forces.cpp             toolbox/clothsimulation/seam_forces.h
        toolbox/clothsimulation/constraint_forces.cpp       toolbox/clothsimulation/constraint_forces.h
        toolbox/clothsimulation/bend_quadratic_forces.cpp   toolbox/clothsimulation/bend_quadratic_forces.h
        toolbox/clothsimulation/stretch_shear_forces.cpp    toolbox/clothsimulation/stretch_shear_forces.h
		toolbox/clothsimulation/cloth.cpp			        toolbox/clothsimulation/cloth.h
        toolbox/clothsimulation/rest_shape.cpp              toolbox/clothsimulation/rest_shape.h

        toolbox/garmentcreation/garment_boundaries.cpp      toolbox/garmentcreation/garment_boundaries.h
        toolbox/garmentcreation/smooth_boundaries.cpp       toolbox/garmentcreation/smooth_boundaries.h
        toolbox/garmentcreation/dijkstra.cpp                toolbox/garmentcreation/dijkstra.h
        toolbox/garmentcreation/connectedSubmesh.cpp        toolbox/garmentcreation/connectedSubmesh.h
        toolbox/garmentcreation/remesh.cpp                  toolbox/garmentcreation/remesh.h
)
add_executable(${PROJECT_NAME} ${SRCFILES})

target_link_libraries(${PROJECT_NAME}
    igl::core
    igl::opengl
    igl::opengl_glfw
    igl::opengl_glfw_imgui
    igl::png
    OpenMP::OpenMP_CXX
    pmp
)

# Cholmod
if(UNIX AND NOT APPLE)
    include_directories("/home/katja/Desktop/SuiteSparse/include/")
    add_definitions(-DHAVE_CHOLMOD)
    target_link_libraries(${PROJECT_NAME} /home/katja/Desktop/SuiteSparse/lib/libcholmod.so)
endif()

# deal with experimental_filesystem
if(WIN32)
    target_compile_options(${PROJECT_NAME} PUBLIC /D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
else()
	target_compile_options(${PROJECT_NAME} PUBLIC)
endif()


# fix cut meshes extra executable
add_executable("fix_cut_meshes" fix_cut_meshes/main.cpp)
target_link_libraries("fix_cut_meshes" igl::core)


