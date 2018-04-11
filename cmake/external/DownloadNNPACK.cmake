cmake_minimum_required(VERSION 2.8.12)

project(nnpack-download NONE)

include(ExternalProject)

ExternalProject_Add(nnpack
                    GIT_REPOSITORY "https://github.com/Maratyszcza/NNPACK.git"
                    GIT_TAG "master"
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/nnpack"
                    BINARY_DIR "${CMAKE_BINARY_DIR}/nnpack-build"
                    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/third_party/nnpack/build"
                               "-DCMAKE_BUILD_TYPE=Release"
                               "-DNNPACK_LIBRARY_TYPE=static"
                               "-DNNPACK_BUILD_TESTS=OFF"
                    BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target install
                    INSTALL_COMMAND ""
                    TEST_COMMAND ""
                    )
