from __future__ import print_function

from shadow.util import mkdir_p
from caffe2shadow import caffe2shadow
from mxnet2shadow import mxnet2shadow

import config_caffe
import config_mxnet


if __name__ == '__main__':
    meta_net_info = config_mxnet.meta_net_mtcnn_info
    model_root = 'model_caffe'
    save_root = model_root + '/shadow'
    copy_params = False
    save_split = True

    shadow_net = caffe2shadow(model_root, meta_net_info, copy_params)

    mkdir_p(save_root)
    if save_split:
        for n, model_name in enumerate(meta_net_info['model_name']):
            save_name = save_root + '/' + model_name
            if copy_params:
                shadow_net.write_proto_to_binary(save_name + '.shadowmodel', n)
                print('Convert successful, model has been written to ' + save_name + '.shadowmodel')
            else:
                shadow_net.write_proto_to_txt(save_name + '.shadowtxt', n)
                print('Convert successful, model has been written to ' + save_name + '.shadowtxt')
    else:
        save_name = save_root + '/' + meta_net_info['save_name']
        if copy_params:
            shadow_net.write_proto_to_binary(save_name + '.shadowmodel')
            print('Convert successful, model has been written to ' + save_name + '.shadowmodel')
        else:
            shadow_net.write_proto_to_txt(save_name + '.shadowtxt')
            print('Convert successful, model has been written to ' + save_name + '.shadowtxt')
