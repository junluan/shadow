#!/bin/bash

set -ex

SHADOW_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
SHADOW_BUILD_ROOT=$SHADOW_ROOT/build

######################### Building Shadow #########################
echo "============================================================"
echo "Building Shadow ... "
echo "Building directory $SHADOW_BUILD_ROOT"
echo "============================================================"
if ! [ -d $SHADOW_BUILD_ROOT ]; then
  mkdir $SHADOW_BUILD_ROOT
fi
cd $SHADOW_BUILD_ROOT
SHADOW_CMAKE_ARGS=('-DCMAKE_INSTALL_PREFIX=.')
SHADOW_CMAKE_ARGS+=('-DCMAKE_BUILD_TYPE=Release')
if [ "$USE_CUDA" = 'true' ]; then
    SHADOW_CMAKE_ARGS+=('-DUSE_CUDA=ON')
else
    SHADOW_CMAKE_ARGS+=('-DUSE_CUDA=OFF')
fi
if [ "$USE_CUDNN" = 'true' ]; then
    SHADOW_CMAKE_ARGS+=('-DUSE_CUDNN=ON')
else
    SHADOW_CMAKE_ARGS+=('-DUSE_CUDNN=OFF')
fi
if [ "$USE_Eigen" = 'true' ]; then
    SHADOW_CMAKE_ARGS+=('-DUSE_Eigen=ON')
else
    SHADOW_CMAKE_ARGS+=('-DUSE_Eigen=OFF')
fi
if [ "$USE_BLAS" = 'true' ]; then
    SHADOW_CMAKE_ARGS+=('-DUSE_BLAS=ON')
else
    SHADOW_CMAKE_ARGS+=('-DUSE_BLAS=OFF')
fi
if [ "$USE_NNPACK" = 'true' ]; then
    SHADOW_CMAKE_ARGS+=('-DUSE_NNPACK=ON')
else
    SHADOW_CMAKE_ARGS+=('-DUSE_NNPACK=OFF')
fi
if [ "$BUILD_SHARED_LIBS" = 'true' ]; then
    SHADOW_CMAKE_ARGS+=('-DBUILD_SHARED_LIBS=ON')
else
    SHADOW_CMAKE_ARGS+=('-DBUILD_SHARED_LIBS=OFF')
fi
cmake .. ${SHADOW_CMAKE_ARGS[*]}
if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    make "-j$(nproc)" install
elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
    make "-j$(sysctl -n hw.ncpu)" install
fi
