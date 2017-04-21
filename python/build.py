from __future__ import print_function

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/shadow')
import util as util


def build(args):
    use_cuda = 'cuda' in args.features
    use_cl = 'cl' in args.features
    use_cudnn = 'cudnn' in args.features
    use_blas = 'openblas' in args.features
    use_nnpack = 'nnpack' in args.features
    use_protobuf = 'protobuf' in args.features
    use_glog = 'glog' in args.features
    use_opencv = 'opencv' in args.features
    build_test = 'test' in args.features

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
    else:
        cmake_options.append('-DUSE_GLog=OFF')
    if use_opencv:
        cmake_options.append('-DUSE_OpenCV=ON')
    else:
        cmake_options.append('-DUSE_OpenCV=OFF')
    if build_test:
        cmake_options.append('-DBUILD_TEST=ON')
    else:
        cmake_options.append('-DBUILD_TEST=OFF')

    shadow_root = os.path.dirname(os.path.abspath(__file__)) + '/..'
    build_root = shadow_root + '/build/' + args.subdir

    if not os.path.isdir(build_root):
        util.mkdir_p(build_root)

    shell_cmd = 'cd ' + build_root + ' && '
    shell_cmd += 'cmake ../.. -DCMAKE_BUILD_TYPE=Release '
    for op in cmake_options:
        shell_cmd += op + ' '
    if args.generator == 'make':
        shell_cmd += '&& make -j2'
    elif args.generator == 'ninja':
        shell_cmd += '-GNinja && ninja -j2'

    if args.debug != 'nodebug':
        print(shell_cmd)

    subprocess.check_call(shell_cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build files!')
    parser.add_argument('--subdir', '-d', default='default',
                        help='The subdirectory for building which is relevant to build.')
    parser.add_argument('--features', '-f', nargs='*', default=[],
                        help='Enable features to build.')
    parser.add_argument('--generator', '-g', default='make',
                        help='The cmake generators, default is gnu make.')
    parser.add_argument('--debug', '-D', nargs='?', const='debug', default='nodebug',
                        help='Open debug mode.')
    args = parser.parse_args()

    build(args)
