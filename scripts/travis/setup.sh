#!/bin/bash
# This script should be sourced, not executed

set -e

export USE_CUDA=true
export USE_CUDNN=false
export USE_Eigen=true
export USE_BLAS=false
export USE_NNPACK=true
export USE_Protobuf=true
export USE_JSON=false
export USE_OpenCV=true
export BUILD_SHARED_LIBS=true

if [ "$BUILD" = 'linux' ]; then
    export USE_CUDA=false
    export USE_NNPACK=false
elif [ "$BUILD" = 'linux-cuda' ]; then
    :
elif [ "$BUILD" = 'linux-cuda-cudnn' ]; then
    export USE_CUDNN=true
elif [ "$BUILD" = 'osx' ]; then
    # Since Python 2.7.14, HomeBrew does not link python and pip in /usr/local/bin/,
    # but they are available in /usr/local/opt/python/libexec/bin/
    export PATH="/usr/local/opt/python/libexec/bin:${PATH}"
else
    echo "BUILD \"$BUILD\" is unknown"
    exit 1
fi
