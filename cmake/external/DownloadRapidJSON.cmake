cmake_minimum_required(VERSION 2.8.12)

project(rapidjson-download NONE)

include(ExternalProject)

ExternalProject_Add(rapidjson
                    GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
                    GIT_TAG "master"
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/rapidjson"
                    BINARY_DIR "${CMAKE_BINARY_DIR}/rapidjson-build"
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    TEST_COMMAND ""
                    )
