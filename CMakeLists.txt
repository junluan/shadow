cmake_minimum_required(VERSION 3.2)
project(shadow)

option(USE_CUDA "Using CUDA!" ON)
option(USE_CL "Using OpenCL!" OFF)
option(USE_OpenCV "Using OpenCV to read, write and show image!" ON)
option(VERBOSE "Showing verbose message" ON)

set(OpenCV_ROOT "/usr/local" CACHE STRING "Give the root of OpenCV!")

set(CMAKE_CXX_COMPILER "/usr/bin/clang++-3.8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (${VERBOSE} STREQUAL "ON")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERBOSE")
endif ()

if (${USE_OpenCV} STREQUAL "ON")
  find_package(OpenCV PATHS ${OpenCV_ROOT} NO_DEFAULT_PATH)
  if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "Can't find OpenCV, give the OpenCV_ROOT!")
  endif ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OpenCV")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif ()

include_directories("./inc" "./inc/layers" "./inc/util" "./external/arcsoft" "./external/jsoncpp" "./external/stb")
file(GLOB shadow_lib_src "./src/network.cpp" "./src/layers/*.cpp" "./src/util/*.cpp" "./external/jsoncpp/jsoncpp.cpp")
file(GLOB shadow_lib_inc "inc/network.hpp" "./inc/layers/*.hpp" "./inc/util/*.hpp")

if (${USE_CUDA} STREQUAL "ON" AND ${USE_CL} STREQUAL "OFF")
  find_package(CUDA REQUIRED)
  if (NOT CUDA_FOUND)
    message(FATAL_ERROR "Can't find CUDA")
  endif ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
  set(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=\"compute_20,sm_20\"")

  cuda_add_library(shadow "./src/util/kernel.cu" ${shadow_lib_src} ${shadow_lib_inc})

  file(GLOB yolo_src "./src/main.cpp" "./src/yolo.cpp" "inc/yolo.hpp")
  cuda_add_executable(yolo ${yolo_src})
  target_link_libraries(yolo shadow ${CUDA_cublas_LIBRARY} ${OpenCV_LIBS})
endif ()

if (${USE_CUDA} STREQUAL "OFF" AND ${USE_CL} STREQUAL "ON")
  find_package(OpenCL REQUIRED)
  if (NOT OpenCL_FOUND)
    message(FATAL_ERROR "Can't find OpenCL")
  endif ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CL")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${PROJECT_SOURCE_DIR}/external/EasyCL/dist/lib")
  include_directories("${PROJECT_SOURCE_DIR}/external/EasyCL/dist/include/easycl")
  include_directories(${OpenCL_INCLUDE_DIRS})

  add_library(shadow ${shadow_lib_src} ${shadow_lib_inc})

  file(GLOB yolo_src "./src/main.cpp" "./src/yolo.cpp" "inc/yolo.hpp")
  add_executable(yolo ${yolo_src})
  target_link_libraries(yolo shadow clBLAS clew EasyCL ${OpenCL_LIBRARIES} ${OpenCV_LIBS})
endif ()

if (${USE_CUDA} STREQUAL "OFF" AND ${USE_CL} STREQUAL "OFF")
  add_library(shadow ${shadow_lib_src} ${shadow_lib_inc})

  file(GLOB yolo_src "./src/main.cpp" "./src/yolo.cpp" "inc/yolo.hpp")
  add_executable(yolo ${yolo_src})
  target_link_libraries(yolo shadow ${OpenCV_LIBS})
endif ()
