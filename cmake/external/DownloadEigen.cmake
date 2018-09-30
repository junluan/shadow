cmake_minimum_required(VERSION 2.8.12)

project(eigen-download NONE)

include(ExternalProject)

ExternalProject_Add(eigen
                    GIT_REPOSITORY "https://github.com/eigenteam/eigen-git-mirror.git"
                    GIT_TAG "3.3.5"
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/eigen3"
                    BINARY_DIR "${CMAKE_BINARY_DIR}/eigen-build"
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    TEST_COMMAND ""
                    )
