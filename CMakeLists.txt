cmake_minimum_required(VERSION 3.4)
project(shadow)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(shadow ${SOURCE_FILES})