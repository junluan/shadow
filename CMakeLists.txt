cmake_minimum_required(VERSION 3.2)
project(shadow)

option(USE_CUDA "Use CUDA" ON)
option(USE_CL "Use OpenCL" OFF)

option(USE_CUDNN "Use CUDNN for acceleration" OFF)

option(USE_Eigen "Use Eigen in CPU mode" ON)
option(USE_BLAS "Use BLAS library in CPU mode" OFF)
option(USE_NNPACK "Use NNPACK library in CPU mode" ON)

option(USE_Protobuf "Use Protobuf" ON)
option(USE_GLog "Use GLog" OFF)
option(USE_OpenCV "Use OpenCV to read, write and show image" ON)

option(BUILD_SHARED_LIBS "Build shared library" ON)

option(BUILD_TEST "Build gtest to test shadow" OFF)

set(BLAS "Open" CACHE STRING "Selected BLAS library")
set(OpenCV_DIR "/usr/local/opencv-3.2.0" CACHE PATH "OpenCV root directory")

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/distribute)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Release Debug." FORCE)
endif ()

if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
else ()
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4005 /wd4018 /wd4244 /wd4267 /wd4305 /wd4309 /wd4800 /wd4819 /wd4996")
endif ()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Utils.cmake)
include(cmake/Dependencies.cmake)

include_directories("third_party" "third_party/arcsoft/include")

add_subdirectory(shadow)

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
include(CPack)
