#!/bin/bash

set -e
set -x

ROOT_DIR="$(cd "$(dirname "$0")/../" && pwd)"
cd $ROOT_DIR

APT_INSTALL_CMD='sudo apt-get install -y --no-install-recommends --allow-unauthenticated'

if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    sudo apt-get update
    $APT_INSTALL_CMD \
        build-essential \
        cmake \
        libopencv-dev \
        libprotobuf-dev \
        protobuf-compiler

    if [ "$BUILD_CUDA" = 'true' ]; then
        CUDA_REPO_PKG='cuda-repo-ubuntu1404_8.0.61-1_amd64.deb'
        CUDA_PKG_VERSION='8-0'
        CUDA_VERSION='8.0'
        wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/${CUDA_REPO_PKG}"
        sudo dpkg -i "$CUDA_REPO_PKG"
        rm -f "$CUDA_REPO_PKG"
        sudo apt-get update
        $APT_INSTALL_CMD \
            "cuda-core-${CUDA_PKG_VERSION}" \
            "cuda-cublas-dev-${CUDA_PKG_VERSION}" \
            "cuda-cudart-dev-${CUDA_PKG_VERSION}" \
            "cuda-curand-dev-${CUDA_PKG_VERSION}" \
            "cuda-driver-dev-${CUDA_PKG_VERSION}" \
            "cuda-nvrtc-dev-${CUDA_PKG_VERSION}"
        sudo ln -sf /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
    fi
    if [ "$BUILD_CUDNN" = 'true' ]; then
        CUDNN_REPO_PKG='nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb'
        CUDNN_PKG_VERSION='7.0.5.15-1+cuda8.0'
        wget "https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/${CUDNN_REPO_PKG}"
        sudo dpkg -i "$CUDNN_REPO_PKG"
        rm -f "$CUDNN_REPO_PKG"
        sudo apt-get update
        $APT_INSTALL_CMD \
            "libcudnn7=${CUDNN_PKG_VERSION}" \
            "libcudnn7-dev=${CUDNN_PKG_VERSION}"
    fi
elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
    brew update
    brew install python || brew upgrade python
    pip uninstall -y numpy  # use brew version (opencv dependency)
    brew install opencv || brew upgrade opencv
    brew install protobuf || brew upgrade protobuf
else
    echo "OS \"$TRAVIS_OS_NAME\" is unknown"
    exit 1
fi
