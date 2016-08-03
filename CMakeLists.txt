cmake_minimum_required(VERSION 3.2)
project(shadow)

option(USE_CUDA "Using CUDA!" ON)
option(USE_CL "Using OpenCL!" OFF)
option(VERBOSE "Showing verbose message" ON)
option(USE_GLog "Using GLog!" ON)
option(USE_OpenCV "Using OpenCV to read, write and show image!" ON)
set(OpenCV_ROOT "/usr/local" CACHE STRING "Give the root of OpenCV!")

option(BUILD_TEST "Build gtest to test layers!" ON)

set(CMAKE_CXX_COMPILER "/usr/bin/clang++-3.8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${PROJECT_SOURCE_DIR}/external/lib")

if (${VERBOSE} STREQUAL "ON")
  add_definitions(-DVERBOSE)
endif ()

if (${USE_GLog} STREQUAL "ON")
  add_definitions(-DUSE_GLog)
endif ()

if (${USE_OpenCV} STREQUAL "ON")
  find_package(OpenCV PATHS ${OpenCV_ROOT} NO_DEFAULT_PATH)
  if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "Can't find OpenCV, give the OpenCV_ROOT!")
  endif ()
  add_definitions(-DUSE_OpenCV)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif ()

include_directories("inc" "external")
file(GLOB shadow_lib_src "src/shadow/*.cpp" "src/shadow/layers/*.cpp" "src/shadow/util/*.cpp" "src/shadow/proto/*.cc")

if (BUILD_TEST)
  file(GLOB test_shadow_src "src/test/*.cpp")
  add_executable(test_shadow ${test_shadow_src})
endif ()

if (${USE_CUDA} STREQUAL "ON" AND ${USE_CL} STREQUAL "OFF")
  find_package(CUDA REQUIRED)
  if (NOT CUDA_FOUND)
    message(FATAL_ERROR "Can't find CUDA")
  endif ()
  add_definitions(-DUSE_CUDA)
  set(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=\"compute_20,sm_20\"")

  cuda_add_library(shadow "src/shadow/kernel.cu" ${shadow_lib_src})

  if (BUILD_TEST)
    target_link_libraries(test_shadow shadow glog protobuf gtest ${CUDA_cublas_LIBRARY} ${OpenCV_LIBS})
  endif ()

  file(GLOB yolo_src "src/main.cpp" "src/yolo.cpp")
  add_executable(yolo ${yolo_src})
  target_link_libraries(yolo shadow glog protobuf ${CUDA_cublas_LIBRARY} ${OpenCV_LIBS})
endif ()

if (${USE_CUDA} STREQUAL "OFF" AND ${USE_CL} STREQUAL "ON")
  find_package(OpenCL REQUIRED)
  if (NOT OpenCL_FOUND)
    message(FATAL_ERROR "Can't find OpenCL")
  endif ()
  add_definitions(-DUSE_CL)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${PROJECT_SOURCE_DIR}/external/EasyCL/dist/lib")
  include_directories("${PROJECT_SOURCE_DIR}/external/EasyCL/dist/include/easycl")
  include_directories(${OpenCL_INCLUDE_DIRS})

  add_library(shadow ${shadow_lib_src})

  if (BUILD_TEST)
    target_link_libraries(test_shadow shadow glog protobuf gtest clBLAS clew EasyCL ${OpenCL_LIBRARIES} ${OpenCV_LIBS})
  endif ()

  file(GLOB yolo_src "src/main.cpp" "src/yolo.cpp")
  add_executable(yolo ${yolo_src})
  target_link_libraries(yolo shadow glog protobuf clBLAS clew EasyCL ${OpenCL_LIBRARIES} ${OpenCV_LIBS})
endif ()

if (${USE_CUDA} STREQUAL "OFF" AND ${USE_CL} STREQUAL "OFF")
  add_library(shadow ${shadow_lib_src})

  if (BUILD_TEST)
    target_link_libraries(test_shadow shadow glog protobuf gtest ${OpenCV_LIBS})
  endif ()

  file(GLOB yolo_src "src/main.cpp" "src/yolo.cpp")
  add_executable(yolo ${yolo_src})
  target_link_libraries(yolo shadow glog protobuf ${OpenCV_LIBS})
endif ()
