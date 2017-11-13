################### Inception-BN info ###################
net_inception_bn = {
    'num_class': [1000],
    'input_name': ['data'],
    'input_shape': [[1, 3, 224, 224]],
    'mean_value': [],
    'scale': 1,
    'out_blob': []
}

meta_net_inception_bn_info = {
    'version': '0.0.1',
    'method': 'classification',
    'model_name': ['Inception-BN'],
    'model_epoch': [126],
    'save_name': 'Inception-BN',
    'network': [net_inception_bn]
}

################### squeezenet info ###################
net_squeezenet = {
    'num_class': [1000],
    'input_name': ['data'],
    'input_shape': [[1, 3, 224, 224]],
    'mean_value': [],
    'scale': 1,
    'out_blob': []
}

meta_net_squeezenet_info = {
    'version': '0.0.1',
    'method': 'classification',
    'model_name': ['squeezenet_v1.1'],
    'model_epoch': [0],
    'save_name': 'squeezenet_v1.1',
    'network': [net_squeezenet]
}

################### dcn_rfcn info ###################
net_dcn_rfcn = {
    'num_class': [7],
    'input_name': ['data', 'im_info'],
    'input_shape': [[1, 3, 224, 224], [1, 3]],
    'mean_value': [123.15, 115.90, 103.06],
    'scale': 1,
    'arg': {
        'is_bgr_s_i': False,
        'class_agnostic_s_i': True
    },
    'out_blob': ['rois', 'cls_prob', 'bbox_pred']
}

meta_net_dcn_rfcn_info = {
    'version': '0.0.1',
    'method': 'faster',
    'model_name': ['dcn_rfcn'],
    'model_epoch': [0],
    'save_name': 'dcn_rfcn',
    'network': [net_dcn_rfcn]
}
