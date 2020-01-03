cmake_minimum_required(VERSION 3.5)
project(shadow)

option(USE_CUDA "Use CUDA" ON)
option(USE_CUDNN "Use CUDNN for acceleration" ON)

option(USE_Eigen "Use Eigen in CPU mode" ON)
option(USE_BLAS "Use BLAS library in CPU mode" OFF)
option(USE_NNPACK "Use NNPACK library in CPU mode" ON)
option(USE_DNNL "Use DNNL library in CPU mode" OFF)

option(USE_Protobuf "Use Protobuf" ON)
option(USE_JSON "Use JSON" OFF)
option(USE_OpenCV "Use OpenCV to read, write and show image" ON)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_SERVICE "Build grpc server and client" OFF)
option(BUILD_LINT "Build clang-format lint" OFF)

option(BUILD_SHARED_LIBS "Build shared library" ON)

set(BLAS "OpenBLAS" CACHE STRING "Selected BLAS library in CPU mode")
set_property(CACHE BLAS PROPERTY STRINGS OpenBLAS MKL)
set(OpenCV_DIR "/usr/local" CACHE PATH "OpenCV root directory")

if (NOT CMAKE_INSTALL_PREFIX)
  message(STATUS "CMake install prefix not set - defaulting to ${CMAKE_BINARY_DIR}/distribute")
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/distribute)
endif ()

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Release Debug." FORCE)
endif ()

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif ()

list(APPEND CMAKE_FIND_ROOT_PATH ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Utils.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(shadow)

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
include(CPack)
