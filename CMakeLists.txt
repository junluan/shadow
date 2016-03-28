cmake_minimum_required(VERSION 3.4)
project(shadow)


set(CMAKE_CXX_COMPILER "/usr/bin/clang++-3.8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DUSE_CL -DVERBOSE -I/usr/local/opencv-3.1.0/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${PROJECT_SOURCE_DIR}/external/EasyCL/dist/include/easycl")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL -lclBLAS -lpthread" )
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -L/usr/local/opencv-3.1.0/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lclew -lEasyCL -L${PROJECT_SOURCE_DIR}/external/EasyCL/dist/lib")

file(GLOB shadow_lib_src "./src/network.cpp" "./src/layers/*.cpp" "./src/util/*.cpp")
file(GLOB shadow_lib_include "./include/network.h" "./include/layers/*.h" "./include/util/*.h")
include_directories("./include" "./include/layers" "./include/util")

add_library(shadow ${shadow_lib_src} ${shadow_lib_include})

add_executable(yolo "./src/main.cpp" "./src/yolo.cpp" "./include/yolo.h")

target_link_libraries(yolo shadow)