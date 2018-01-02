from __future__ import print_function

from shadow.util import mkdir_p
from config.config_customs import *
from config.config_examples import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Convert model to shadow!')
    parser.add_argument('--model_root', '-d', required=True, help='The root folder of the models to be converted.')
    parser.add_argument('--config_name', '-c', required=True, help='The suffix name of the model config function.')
    parser.add_argument('--save_root', '-s', default='model_shadow', help='The root folder to save the shadow model.')
    parser.add_argument('--copy_params', '-p', action='store_true', help='Copy source model weights.')
    parser.add_argument('--merge_op', '-m', action='store_true', help='Merge operators.')
    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    args = parse_args()

    meta_net_info = eval('get_config_' + args.config_name)()

    mkdir_p(args.save_root)

    model_type = meta_net_info['model_type']
    if model_type == 'caffe':
        from shadow.caffe2shadow import caffe2shadow
        shadow_net = caffe2shadow(args.model_root, meta_net_info, args.copy_params)
    elif model_type == 'mxnet':
        from shadow.mxnet2shadow import mxnet2shadow
        shadow_net = mxnet2shadow(args.model_root, meta_net_info, args.copy_params)
    else:
        raise ValueError('Currently only support convert caffe or mxnet model!', model_type)

    if args.merge_op:
        from shadow.merger import Merge
        merged_net = Merge(shadow_net, args.copy_params)

    if args.copy_params:
        for n, model_name in enumerate(meta_net_info['model_name']):
            save_path = args.save_root + '/' + model_name + '.shadowmodel'
            shadow_net.write_proto_to_binary(save_path, n)
            print('Convert successful, model has been written to ' + save_path)
            if args.merge_op:
                save_path = args.save_root + '/' + model_name + '_merged.shadowmodel'
                merged_net.write_proto_to_binary(save_path, n)
                print('Convert successful, merged model has been written to ' + save_path)
    else:
        for n, model_name in enumerate(meta_net_info['model_name']):
            save_path = args.save_root + '/' + model_name + '.shadowtxt'
            shadow_net.write_proto_to_txt(save_path, n)
            print('Convert successful, model has been written to ' + save_path)
            if args.merge_op:
                save_path = args.save_root + '/' + model_name + '_merged.shadowtxt'
                merged_net.write_proto_to_txt(save_path, n)
                print('Convert successful, merged model has been written to ' + save_path)
