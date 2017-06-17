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
    use_eigen = 'eigen' in args.features
    use_blas = 'openblas' in args.features
    use_nnpack = 'nnpack' in args.features
    use_protobuf = 'protobuf' in args.features
    use_glog = 'glog' in args.features
    use_opencv = 'opencv' in args.features
    build_shared_libs = 'shared' in args.features
    build_examples = 'examples' in args.features
    build_tools = 'tools' in args.features
    build_service = 'service' in args.features
    build_test = 'test' in args.features

    cmake_options = {}
    if use_cuda:
        cmake_options['USE_CUDA'] = True
        cmake_options['USE_CL'] = False
        cmake_options['USE_CUDNN'] = use_cudnn
    elif use_cl:
        cmake_options['USE_CUDA'] = False
        cmake_options['USE_CL'] = True
    else:
        cmake_options['USE_CUDA'] = False
        cmake_options['USE_CL'] = False
        cmake_options['USE_Eigen'] = use_eigen
        cmake_options['USE_BLAS'] = use_blas
        cmake_options['USE_NNPACK'] = use_nnpack
    cmake_options['USE_Protobuf'] = use_protobuf
    cmake_options['USE_GLog'] = use_glog
    cmake_options['USE_OpenCV'] = use_opencv
    cmake_options['BUILD_SHARED_LIBS'] = build_shared_libs
    cmake_options['BUILD_EXAMPLES'] = build_examples
    cmake_options['BUILD_TOOLS'] = build_tools
    cmake_options['BUILD_SERVICE'] = build_service
    cmake_options['BUILD_TEST'] = build_test

    shadow_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'
    build_root = shadow_root + '/build/' + args.subdir

    if not os.path.isdir(build_root):
        util.mkdir_p(build_root)

    shell_cmd = 'cd ' + build_root + ' && '
    shell_cmd += 'cmake ../.. '
    choices = ['OFF', 'ON']
    for define in args.define:
        shell_cmd += '-D' + define + ' '
    for (feature, value) in cmake_options.items():
        shell_cmd += '-D' + feature + '=' + choices[int(value)] + ' '
    if args.generator == 'make':
        shell_cmd += '&& make -j2'
    elif args.generator == 'ninja':
        shell_cmd += '-GNinja && ninja'

    if args.debug != 'nodebug':
        print(shell_cmd + '\n' + ''.join(['='] * 60))

    subprocess.check_call(shell_cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build files!')
    parser.add_argument('--subdir', '-d', default='default',
                        help='The subdirectory for building, which is relevant to build.')
    parser.add_argument('--features', '-f', nargs='*', default=[],
                        help='Enable features to build.')
    parser.add_argument('--define', '-D', nargs='+', default=[],
                        help='Other flags should be passed to cmake.')
    parser.add_argument('--generator', '-g', default='make',
                        help='The cmake generators, default is gnu make.')
    parser.add_argument('--debug', nargs='?', const='debug', default='nodebug',
                        help='Open debug mode.')
    args = parser.parse_args()

    build(args)
