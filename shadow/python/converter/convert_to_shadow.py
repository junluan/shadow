from config import *
from shadow import converter
from shadow import optimizer

import argparse
import os


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

    os.makedirs(args.save_root, exist_ok=True)

    meta_net_info = eval('get_config_' + args.config_name)()

    network = converter.convert(args.model_root, meta_net_info, args.copy_params)

    if args.merge_op:
        optimizer.optimize(network)

    save_name = '{}/{}'.format(args.save_root, meta_net_info['save_name'])

    if args.copy_params:
        save_path = save_name + ('_merged.shadowmodel' if args.merge_op else '.shadowmodel')
        network.write_proto_to_binary(save_path)
    else:
        save_path = save_name + ('_merged.shadowtxt' if args.merge_op else '.shadowtxt')
        network.clear_all_blobs()
        network.write_proto_to_txt(save_path)

    print('Convert successful, model has been written to ' + save_path)
