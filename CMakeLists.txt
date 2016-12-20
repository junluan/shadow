cmake_minimum_required(VERSION 3.2)
project(shadow)

option(USE_CUDA "Use CUDA" ON)
option(USE_CL "Use OpenCL" OFF)
option(USE_BLAS "Use BLAS library in CPU mode" ON)
option(USE_NNPACK "Use NNPACK library in CPU mode" ON)
option(USE_GLog "Use GLog" OFF)
option(USE_OpenCV "Use OpenCV to read, write and show image" ON)

option(BUILD_TEST "Build gtest to test shadow!" OFF)

set(BLAS "Open" CACHE STRING "Selected BLAS library")
set(OpenCV_ROOT "/usr/local/opencv-3.1.0" CACHE STRING "Give the root of OpenCV")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")

include_directories("inc" "src/shadow/proto" "external")
link_directories("external/libs")

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tools)
