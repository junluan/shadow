#!/bin/bash

set -ex

if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    CUDA_REPO_PKG="cuda-repo-ubuntu1604_10.0.130-1_amd64.deb"
    CUDA_PKG_VERSION="10-0"
    CUDA_VERSION="10.0"
    wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/$CUDA_REPO_PKG"
    sudo dpkg -i "$CUDA_REPO_PKG"

    CUDNN_REPO_PKG="nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb"
    CUDNN_PKG_VERSION="7.6.5.32-1+cuda10.0"
    wget "https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/$CUDNN_REPO_PKG"
    sudo dpkg -i "$CUDNN_REPO_PKG"

    sudo apt-get update

    APT_INSTALL_CMD="sudo apt-get install -y --no-install-recommends --allow-unauthenticated"

    $APT_INSTALL_CMD \
        cmake \
        build-essential \
        libopencv-dev \
        libprotobuf-dev \
        protobuf-compiler

    $APT_INSTALL_CMD \
        "cuda-compiler-$CUDA_PKG_VERSION" \
        "cuda-cudart-dev-$CUDA_PKG_VERSION" \
        "cuda-cublas-dev-$CUDA_PKG_VERSION" \
        "libcudnn7=$CUDNN_PKG_VERSION" \
        "libcudnn7-dev=$CUDNN_PKG_VERSION"

    sudo ln -sf /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    brew update
    brew install --force --ignore-dependencies opencv
fi
