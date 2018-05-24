cmake_minimum_required(VERSION 2.8.12)

project(googletest-download NONE)

include(ExternalProject)

ExternalProject_Add(googletest
                    GIT_REPOSITORY "https://github.com/google/googletest.git"
                    GIT_TAG "master"
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/googletest"
                    BINARY_DIR "${CMAKE_BINARY_DIR}/googletest-build"
                    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/third_party/googletest/build"
                               "-DCMAKE_BUILD_TYPE=Release"
                    BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target install
                    INSTALL_COMMAND ""
                    TEST_COMMAND ""
                    )
