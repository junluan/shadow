#!/bin/bash

set -ex

SHADOW_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
SHADOW_BUILD_ROOT=$SHADOW_ROOT/build

######################### Building Shadow #########################
echo "============================================================"
echo "Building Shadow ... "
echo "Building directory $SHADOW_BUILD_ROOT"
echo "============================================================"

mkdir -p "$SHADOW_BUILD_ROOT" && cd "$SHADOW_BUILD_ROOT"

CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$SHADOW_BUILD_ROOT")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DUSE_CUDA=$USE_CUDA")
CMAKE_ARGS+=("-DUSE_CUDNN=$USE_CUDNN")
CMAKE_ARGS+=("-DUSE_Eigen=$USE_Eigen")
CMAKE_ARGS+=("-DUSE_BLAS=$USE_BLAS")
CMAKE_ARGS+=("-DUSE_NNPACK=$USE_NNPACK")
CMAKE_ARGS+=("-DUSE_DNNL=$USE_DNNL")
CMAKE_ARGS+=("-DUSE_Protobuf=$USE_Protobuf")
CMAKE_ARGS+=("-DUSE_JSON=$USE_JSON")
CMAKE_ARGS+=("-DUSE_OpenCV=$USE_OpenCV")
CMAKE_ARGS+=("-DBUILD_EXAMPLES=$BUILD_EXAMPLES")
CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS")

cmake .. ${CMAKE_ARGS[*]}

if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    cmake --build . --target install -- "-j$(nproc)"
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    cmake --build . --target install -- "-j$(sysctl -n hw.ncpu)"
fi
