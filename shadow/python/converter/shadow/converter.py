from .network import Network


def convert(model_root, meta_net_info):
    model_type = meta_net_info['model_type']
    model_name = meta_net_info['model_name']
    model_epoch = meta_net_info['model_epoch']
    net_info = meta_net_info['network']

    assert len(model_type) == len(model_name) == len(net_info)

    network = Network(meta_net_info['save_name'])

    if 'arg' in meta_net_info:
        network.set_meta_net_arg_dict(meta_net_info['arg'])

    for n, type in enumerate(model_type):
        network.set_net(n)
        if type == 'caffe':
            from .convert_caffe import convert_caffe
            convert_caffe(network, net_info[n], model_root, model_name[n])
        elif type == 'mxnet':
            from .convert_mxnet import convert_mxnet
            assert n < len(model_epoch)
            convert_mxnet(network, net_info[n], model_root, model_name[n], model_epoch[n])
        elif type == 'onnx':
            from .convert_onnx import convert_onnx
            convert_onnx(network, net_info[n], model_root, model_name[n])
        else:
            raise ValueError('Currently only support convert caffe, mxnet or onnx model!', type)

    return network
