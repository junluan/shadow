#######################################################
################### squeezenet info ###################
#######################################################
def get_config_squeezenet():
    net_info = {
        'num_class': [1000],
        'input_name': ['data'],
        'input_shape': [[1, 3, 224, 224]],
        'mean_value': [],
        'scale': 1,
        'arg': {},
        'out_blob': []
    }

    meta_net_info = {
        'model_type': 'mxnet',
        'model_name': ['squeezenet_v1.1'],
        'model_epoch': [0],
        'network': [net_info]
    }

    return meta_net_info


#######################################################
##################### mtcnn info ######################
#######################################################
def get_config_mtcnn():
    net_r_info = {
        'num_class': [0],
        'input_name': ['data'],
        'input_shape': [[1, 3, 360, 360]],
        'mean_value': [127.5],
        'scale': 0.0078125,
        'arg': {},
        'out_blob': ['conv4-2', 'prob1']
    }

    net_p_info = {
        'num_class': [0],
        'input_name': ['data'],
        'input_shape': [[50, 3, 24, 24]],
        'mean_value': [127.5],
        'scale': 0.0078125,
        'arg': {},
        'out_blob': ['conv5-2', 'prob1']
    }

    net_o_info = {
        'num_class': [0],
        'input_name': ['data'],
        'input_shape': [[20, 3, 48, 48]],
        'mean_value': [127.5],
        'scale': 0.0078125,
        'arg': {},
        'out_blob': ['conv6-2', 'conv6-3', 'prob1']
    }

    meta_net_info = {
        'model_type': 'caffe',
        'model_name': ['det1', 'det2', 'det3'],
        'model_epoch': [],
        'network': [net_r_info, net_p_info, net_o_info]
    }

    return meta_net_info


#######################################################
###################### ssd info #######################
#######################################################
def get_config_ssd():
    net_info = {
        'num_class': [3],
        'input_name': ['data'],
        'input_shape': [[1, 3, 300, 300]],
        'mean_value': [103.94, 116.78, 123.68],
        'scale': 1,
        'arg': {},
        'out_blob': ['mbox_loc', 'mbox_conf_flatten', 'mbox_priorbox']
    }

    meta_net_info = {
        'model_type': 'caffe',
        'model_name': ['adas_model_finetune_reduce_3'],
        'model_epoch': [],
        'network': [net_info]
    }

    return meta_net_info


#######################################################
################## faster rcnn info ###################
#######################################################
def get_config_faster():
    net_info = {
        'num_class': [21],
        'input_name': ['data'],
        'input_shape': [[1, 3, 224, 224], [1, 3]],
        'mean_value': [102.9801, 115.9465, 122.7717],
        'scale': 1,
        'arg': {
            'is_bgr_s_i': True,
            'class_agnostic_s_i': False
        },
        'out_blob': ['rois', 'cls_prob', 'bbox_pred']
    }

    meta_net_info = {
        'model_type': 'caffe',
        'model_name': ['VGG16_faster_rcnn_final'],
        'model_epoch': [],
        'network': [net_info]
    }

    return meta_net_info
