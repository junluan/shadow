from __future__ import print_function

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/shadow')
import util as util


def build(subdir, features, generator):
    use_cuda = 'cuda' in features
    use_cl = 'cl' in features
    use_cudnn = 'cudnn' in features
    use_blas = 'openblas' in features
    use_nnpack = 'nnpack' in features
    use_protobuf = 'protobuf' in features
    use_glog = 'glog' in features
    use_opencv = 'opencv' in features
    build_test = 'test' in features

    cmake_options = []
    if use_cuda:
        cmake_options.append('-DUSE_CUDA=ON')
        cmake_options.append('-DUSE_CL=OFF')
        if use_cudnn:
            cmake_options.append('-DUSE_CUDNN=ON')
        else:
            cmake_options.append('-DUSE_CUDNN=OFF')
    elif use_cl:
        cmake_options.append('-DUSE_CUDA=OFF')
        cmake_options.append('-DUSE_CL=ON')
    else:
        cmake_options.append('-DUSE_CUDA=OFF')
        cmake_options.append('-DUSE_CL=OFF')
        if use_blas:
            cmake_options.append('-DUSE_BLAS=ON')
        else:
            cmake_options.append('-DUSE_BLAS=OFF')
        if use_nnpack:
            cmake_options.append('-DUSE_NNPACK=ON')
        else:
            cmake_options.append('-DUSE_NNPACK=OFF')
    if use_protobuf:
        cmake_options.append('-DUSE_Protobuf=ON')
    else:
        cmake_options.append('-DUSE_Protobuf=OFF')
    if use_glog:
        cmake_options.append('-DUSE_GLog=ON')
    if use_opencv:
        cmake_options.append('-DUSE_OpenCV=ON')
    else:
        cmake_options.append('-DUSE_OpenCV=OFF')

    shadow_root = os.path.dirname(os.path.abspath(__file__)) + '/..'
    build_root = shadow_root + '/build/' + subdir

    if not os.path.isdir(build_root):
        util.mkdir_p(build_root)

    shell_cmd = 'cd ' + build_root + '\n'
    shell_cmd += 'cmake ../.. -DCMAKE_BUILD_TYPE=Release '
    for op in cmake_options:
        shell_cmd += op + ' '
    if generator == 'ninja':
        shell_cmd += '-GNinja && ninja -j2'
    elif generator == 'make':
        shell_cmd += '&& make -j2'

    print(shell_cmd)
    subprocess.call(shell_cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build files!')
    parser.add_argument('--subdir', default='default',
                        help='The subdirectory for building which is relevant to build.')
    parser.add_argument('--features', nargs='*', default=[],
                        help='Enable features to build.')
    parser.add_argument('--generator', default='ninja',
                        help='The cmake generators, default is ninja.')
    args = parser.parse_args()

    subdir = args.subdir
    features = args.features
    generator = args.generator

    build(subdir, features, generator)
