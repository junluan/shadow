#!/bin/bash
# This script should be sourced, not executed

set -e

export USE_CUDA=OFF
export USE_CUDNN=OFF
export USE_Eigen=ON
export USE_BLAS=OFF
export USE_NNPACK=ON
export USE_DNNL=ON
export USE_Protobuf=ON
export USE_JSON=OFF
export USE_OpenCV=ON
export BUILD_EXAMPLES=ON
export BUILD_SHARED_LIBS=ON

if [ "$BUILD" = "linux-cpu" ]; then
    :
elif [ "$BUILD" = "linux-cuda" ]; then
    export USE_CUDA=ON
elif [ "$BUILD" = "linux-cuda-cudnn" ]; then
    export USE_CUDA=ON
    export USE_CUDNN=ON
elif [ "$BUILD" = "osx" ]; then
    # Since Python 2.7.14, HomeBrew does not link python and pip in /usr/local/bin/,
    # but they are available in /usr/local/opt/python/libexec/bin/
    export PATH="/usr/local/opt/python/libexec/bin:$PATH"
else
    echo "BUILD \"$BUILD\" is unknown"
    exit 1
fi
