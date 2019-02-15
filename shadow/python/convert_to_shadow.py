from __future__ import print_function

from config import *
from shadow import converter
from shadow import merger
from shadow import util

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

    util.mkdir_p(args.save_root)

    meta_net_info = eval('get_config_' + args.config_name)()

    network = converter.convert(args.model_root, meta_net_info, args.copy_params)

    network_merged = merger.merge(network, args.copy_params) if args.merge_op else None

    save_name = args.save_root + '/' + meta_net_info['save_name']

    if args.copy_params:
        save_path = save_name + '.shadowmodel'
        network.write_proto_to_binary(save_path)
        print('Convert successful, model has been written to ' + save_path)
        if network_merged is not None:
            save_path = save_name + '_merged.shadowmodel'
            network_merged.write_proto_to_binary(save_path)
            print('Merge successful, merged model has been written to ' + save_path)
    else:
        save_path = save_name + '.shadowtxt'
        network.write_proto_to_txt(save_path)
        print('Convert successful, model has been written to ' + save_path)
        if network_merged is not None:
            save_path = save_name + '_merged.shadowtxt'
            network_merged.write_proto_to_txt(save_path)
            print('Merge successful, merged model has been written to ' + save_path)
