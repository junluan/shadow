cmake_minimum_required(VERSION 2.8.12)

project(dnnl-download NONE)

include(ExternalProject)

ExternalProject_Add(dnnl
                    GIT_REPOSITORY "https://github.com/oneapi-src/oneDNN.git"
                    GIT_TAG "master"
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/dnnl"
                    BINARY_DIR "${CMAKE_BINARY_DIR}/dnnl-build"
                    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/third_party/dnnl/build"
                               "-DCMAKE_BUILD_TYPE=Release"
                               "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
                               "-DDNNL_LIBRARY_TYPE=STATIC"
                               "-DDNNL_CPU_RUNTIME=SEQ"
                               "-DDNNL_BUILD_EXAMPLES=OFF"
                               "-DDNNL_BUILD_TESTS=OFF"
                    BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target install
                    INSTALL_COMMAND ""
                    TEST_COMMAND ""
                    )
