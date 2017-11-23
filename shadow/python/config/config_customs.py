#######################################################
##################### custom info #####################
#######################################################
def get_config_custom():
    net_info = {
        'num_class': [1000],  # class numbers
        'input_name': ['data'],  # net input blob names
        'input_shape': [[1, 3, 224, 224]],  # net input blob shapes
        'mean_value': [],  # data mean values
        'scale': 1,  # data scale value
        'arg': {},  # net arguments, must end with one of 's_i, s_f, s_s, v_i, v_f, v_s'
        'out_blob': []  # net output blob names
    }

    meta_net_info = {
        'model_type': 'mxnet',  # model type: caffe or mxnet
        'model_name': ['squeezenet_v1.1'],  # model name on disk
        'model_epoch': [0],  # Only for mxnet model
        'network': [net_info]  # networks
    }

    return meta_net_info
