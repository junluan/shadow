from __future__ import print_function

from shadow.util import mkdir_p
from caffe2shadow import caffe2shadow


################## mtcnn info ##################
net_mtcnn_r = {
    'num_class': [0],
    'input_shape': [[1, 3, 360, 360]],
    'mean_value': [127.5],
    'scale': 0.0078125,
    'out_blob': ['conv4-2', 'prob1']
}

net_mtcnn_p = {
    'num_class': [0],
    'input_shape': [[50, 3, 24, 24]],
    'mean_value': [127.5],
    'scale': 0.0078125,
    'out_blob': ['conv5-2', 'prob1']
}

net_mtcnn_o = {
    'num_class': [0],
    'input_shape': [[20, 3, 48, 48]],
    'mean_value': [127.5],
    'scale': 0.0078125,
    'out_blob': ['conv6-2', 'conv6-3', 'prob1']
}

meta_net_mtcnn_info = {
    'version': '0.0.1',
    'method': 'mtcnn',
    'model_name': ['det1', 'det2', 'det3'],
    'save_name': 'mtcnn',
    'network': [net_mtcnn_r, net_mtcnn_p, net_mtcnn_o]
}

################### ssd info ###################
net_ssd = {
    'num_class': [3],
    'input_shape': [[1, 3, 300, 300]],
    'mean_value': [103.94, 116.78, 123.68],
    'scale': 1,
    'out_blob': ['mbox_loc', 'mbox_conf_flatten', 'mbox_priorbox']
}

meta_net_ssd_info = {
    'version': '0.0.1',
    'method': 'ssd',
    'model_name': ['adas_model_finetune_reduce_3'],
    'save_name': 'adas_model_finetune_reduce_3',
    'network': [net_ssd]
}

############### faster rcnn info ###############
net_faster = {
    'num_class': [21],
    'input_shape': [[1, 3, 224, 224], [1, 3]],
    'mean_value': [102.9801, 115.9465, 122.7717],
    'scale': 1,
    'out_blob': ['bbox_pred', 'cls_prob']
}

meta_net_faster_info = {
    'version': '0.0.1',
    'method': 'faster',
    'model_name': ['VGG16_faster_rcnn_final'],
    'save_name': 'VGG16_faster_rcnn_final',
    'network': [net_faster]
}

################### rfcn info ###################
net_rfcn = {
    'num_class': [21],
    'input_shape': [[1, 3, 224, 224], [1, 3]],
    'mean_value': [102.9801, 115.9465, 122.7717],
    'scale': 1,
    'out_blob': ['bbox_pred', 'cls_prob']
}

meta_net_rfcn_info = {
    'version': '0.0.1',
    'method': 'faster',
    'model_name': ['resnet50_rfcn_final'],
    'save_name': 'resnet50_rfcn_final',
    'network': [net_rfcn]
}

#################################################
if __name__ == '__main__':
    meta_net_info = meta_net_mtcnn_info
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
