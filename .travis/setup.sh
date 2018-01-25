#!/bin/bash
# This script should be sourced, not executed

set -e

export BUILD_CUDA=false
export BUILD_CUDNN=false
export BUILD_Eigen=false
export BUILD_BLAS=false
export BUILD_NNPACK=false
export BUILD_Protobuf=true
export BUILD_OpenCV=true
export BUILD_TEST=false
export BUILD_SHARED_LIBS=true

if [ "$BUILD" = 'linux' ]; then
    :
elif [ "$BUILD" = 'linux-cuda' ]; then
    export BUILD_CUDA=true
elif [ "$BUILD" = 'linux-cuda-cudnn' ]; then
    export BUILD_CUDA=true
    export BUILD_CUDNN=true
elif [ "$BUILD" = 'osx' ]; then
    # Since Python 2.7.14, HomeBrew does not link python and pip in /usr/local/bin/,
    # but they are available in /usr/local/opt/python/libexec/bin/
    export PATH="/usr/local/opt/python/libexec/bin:${PATH}"
else
    echo "BUILD \"$BUILD\" is unknown"
    exit 1
fi
