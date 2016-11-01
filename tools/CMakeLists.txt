cmake_minimum_required(VERSION 3.2)
project(tools)

include_directories("." "../inc")

set(CMAKE_CXX_COMPILER "/usr/bin/clang++-3.8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(caffe2shadow "caffe2shadow.cpp" "../src/shadow/util/io.cpp")
target_link_libraries(caffe2shadow glog proto)
