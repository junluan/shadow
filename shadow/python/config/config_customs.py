#######################################################
##################### custom info #####################
#######################################################
def get_config_custom():
    net_info = {
        'input_name': ['data'],  # net input blob names
        'input_shape': [[1, 3, 224, 224]],  # net input blob shapes
        'mean_value': [],  # data mean values
        'scale_value': [1],  # data scale values
        'arg': {  # net arguments, must end with one of 's_i, s_f, s_s, v_i, v_f, v_s'
            'num_classes_s_i': 1000,  # class numbers
            'out_blob_v_s': [],  # net output blob names
            'is_bgr_s_i': True  # other useful arguments
        }
    }

    meta_net_info = {
        'model_type': 'mxnet',  # model type: caffe or mxnet
        'model_name': ['squeezenet_v1.1'],  # model file name on disk
        'model_epoch': [0],  # only for mxnet model
        'save_name': 'squeezenet_v1.1',  # shadow model saved name
        'network': [net_info]  # networks
    }

    return meta_net_info
