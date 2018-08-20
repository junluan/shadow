from __future__ import print_function

import json
import mxnet as mx
from shadow.net_spec import Shadow


mx_ver = [int(v) for v in mx.__version__.split('.')]
mx_ver_int = 10000 * mx_ver[0] + 100 * mx_ver[1] + mx_ver[2]

if mx_ver_int <= 900:
    params_str = 'param'
elif mx_ver_int <= 1300:
    params_str = 'attr'
else:
    params_str = 'attrs'


def copy_weights(arg_params, aux_params, param_dict, shadow_net):
    net_param = shadow_net.net_param
    for shadow_op in net_param.op:
        op_name = shadow_op.name
        if op_name not in param_dict:
            continue
        for n, param_name in enumerate(param_dict[op_name]):
            if param_name in arg_params:
                param_data = arg_params[param_name].asnumpy().flatten()
            elif param_name in aux_params:
                param_data = aux_params[param_name].asnumpy().flatten()
            elif '_bn_scale' in param_name:
                param_data = [1]
            else:
                raise ValueError(param_name + ' not found in arg_params or aux_params')
            shadow_blob = net_param.blob.add()
            shadow_blob.name = op_name + '_weights:{}'.format(n)
            shadow_blob.shape.extend([len(param_data)])
            shadow_blob.data_f.extend(param_data)
            shadow_op.bottom.append(shadow_blob.name)


def parse_param(params, name, type, default_value):
    if name in params:
        if type == 's_i':
            return int(params[name])
        elif type == 's_f':
            return float(params[name])
        elif type == 's_s':
            return params[name]
        elif type == 'v_i':
            return [int(p) for p in params[name].strip()[1:-1].split(',')]
        elif type == 'v_f':
            return [float(p) for p in params[name].strip()[1:-1].split(',')]
        elif type == 'v_s':
            return params[name].strip()[1:-1].split(',')
    else:
        return default_value


def find_inputs(json_nodes, json_name, json_inputs):
    bottom_names, param_names = [], []
    for input_index in json_inputs:
        assert input_index[1] == 0
        input_node = json_nodes[input_index[0]]
        node_op = input_node['op']
        node_name = input_node['name']
        if node_op == 'Dropout' or node_op == 'Activation' or node_op == 'Flatten':
            inner_bottom_names, _ = find_inputs(json_nodes, json_name, input_node['inputs'])
            bottom_names.extend(inner_bottom_names)
        elif node_op != 'null' or json_name not in node_name:
            bottom_names.append(node_name)
        if node_op == 'null' and json_name in node_name:
            param_names.append(node_name)
    return bottom_names, param_names


def find_nodes(json_nodes, json_indexes):
    node_names = []
    for indexes in json_indexes:
        assert indexes[1] == 0
        json_node = json_nodes[indexes[0]]
        node_names.append(json_node['name'])
    return node_names


def convert_input(net_info, shadow_net):
    shadow_inputs = net_info['input_name']
    shadow_shapes = net_info['input_shape']

    assert len(shadow_inputs) == len(shadow_shapes)
    shadow_net.add_input('input', [], shadow_inputs, shadow_shapes)

    num_mean = len(net_info['mean_value'])
    num_scale = len(net_info['scale_value'])
    mean_value = net_info['mean_value'] if num_mean > 0 else None
    scale_value = net_info['scale_value'] if num_scale > 0 else None
    if num_mean > 0 or num_scale > 0:
        max_dim = max(num_mean, num_scale)
        if num_mean == 0 and num_scale > 0:
            mean_value = [0] * max_dim
        elif num_scale == 0 and num_mean > 0:
            scale_value = [1] * max_dim
        elif num_mean == 1 and num_scale > 1:
            mean_value *= max_dim
        elif num_scale == 1 and num_mean > 1:
            scale_value *= max_dim
        assert len(mean_value) == len(scale_value)
        for i in range(0, len(mean_value)):
            mean_value[i] *= -scale_value[i]
        for input_name in shadow_inputs:
            if 'data' in input_name:
                shadow_net.add_scale(input_name, [input_name], [input_name], 1, 1, False, False, scale_value, mean_value)


def convert_activate(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    act_type = parse_param(json_attr, 'act_type', 's_s', 'relu')
    if act_type == 'relu':
        act_type = 'Relu'
    elif act_type == 'sigmoid':
        act_type = 'Sigmoid'
    elif act_type == 'softrelu':
        act_type = 'SoftPlus'
    elif act_type == 'tanh':
        act_type = 'Tanh'
    else:
        raise ValueError('Unsupported activate type', act_type)

    shadow_net.add_activate(json_name, bottom_names, bottom_names, act_type)


def convert_batch_norm(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)
    if len(param_names) > 0:
        mean_name = json_name + '_moving_mean'
        var_name = json_name + '_moving_var'
        scale_name = json_name + '_bn_scale'
        gamma_name = json_name + '_gamma'
        beta_name = json_name + '_beta'
        param_dict[json_name] = [mean_name, var_name, scale_name]
        param_dict[json_name + '_scale'] = [gamma_name, beta_name]

    use_global_stats = parse_param(json_attr, 'use_global_stats', 's_s', 'True') == 'True'
    use_global_stats = True
    scale_bias_term = True
    eps = parse_param(json_attr, 'eps', 's_f', 1e-5)

    shadow_net.add_batch_norm(json_name, bottom_names, [json_name], use_global_stats, eps)
    shadow_net.add_scale(json_name + '_scale', [json_name], [json_name], 1, 1, True, scale_bias_term)


def convert_binary(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    json_op = json_node['op']
    if json_op == '_plus_scalar':
        operation_type = 'Add'
    elif json_op == '_minus_scalar':
        operation_type = 'Sub'
    elif json_op == '_mul_scalar':
        operation_type = 'Mul'
    elif json_op == '_div_scalar':
        operation_type = 'Div'
    else:
        raise ValueError('Unsupported binary type', json_op)

    scalar = parse_param(json_attr, 'scalar', 's_f', 1)

    shadow_net.add_binary(json_name, bottom_names, [json_name], operation_type, scalar)


def convert_concat(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    dim = parse_param(json_attr, 'dim', 's_i', 1)
    num_args = parse_param(json_attr, 'num_args', 's_i', 2)

    shadow_net.add_concat(json_name, bottom_names, [json_name], dim)


def convert_connected(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)
    if len(param_names) > 0:
        param_dict[json_name] = param_names

    num_output = parse_param(json_attr, 'num_hidden', 's_i', 0)
    bias_term = parse_param(json_attr, 'no_bias', 's_s', 'False') != 'True'

    shadow_net.add_connected(json_name, bottom_names, [json_name], num_output, bias_term)


def convert_conv(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)
    if len(param_names) > 0:
        param_dict[json_name] = param_names

    num_output = parse_param(json_attr, 'num_filter', 's_i', 0)
    kernel_size = parse_param(json_attr, 'kernel', 'v_i', [3, 3])
    stride = parse_param(json_attr, 'stride', 'v_i', [1, 1])
    pad = parse_param(json_attr, 'pad', 'v_i', [0, 0])
    dilate = parse_param(json_attr, 'dilate', 'v_i', [1, 1])
    bias_term = parse_param(json_attr, 'no_bias', 's_s', 'False') != 'True'
    group = parse_param(json_attr, 'num_group', 's_i', 1)

    shadow_net.add_conv(json_name, bottom_names, [json_name], num_output, kernel_size[0], stride[0], pad[0], dilate[0], bias_term, group)


def convert_deformable_conv(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)
    if len(param_names) > 0:
        param_dict[json_name] = param_names

    num_output = parse_param(json_attr, 'num_filter', 's_i', 0)
    kernel_size = parse_param(json_attr, 'kernel', 'v_i', [3, 3])
    stride = parse_param(json_attr, 'stride', 'v_i', [1, 1])
    pad = parse_param(json_attr, 'pad', 'v_i', [0, 0])
    dilate = parse_param(json_attr, 'dilate', 'v_i', [1, 1])
    bias_term = parse_param(json_attr, 'no_bias', 's_s', 'False') != 'True'
    group = parse_param(json_attr, 'num_group', 's_i', 1)
    deformable_group = parse_param(json_attr, 'num_deformable_group', 's_i', 1)

    shadow_net.add_deformable_conv(json_name, bottom_names, [json_name], num_output, kernel_size[0], stride[0], pad[0], dilate[0], bias_term, group, deformable_group)


def convert_deformable_psroi_pooling(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    output_dim = parse_param(json_attr, 'output_dim', 's_i', 0)
    group_size = parse_param(json_attr, 'group_size', 's_i', 0)
    pooled_size = parse_param(json_attr, 'pooled_size', 's_i', 0)
    part_size = parse_param(json_attr, 'part_size', 's_i', 0)
    sample_per_part = parse_param(json_attr, 'sample_per_part', 's_i', 1)
    spatial_scale = parse_param(json_attr, 'spatial_scale', 's_f', 0.0625)
    trans_std = parse_param(json_attr, 'trans_std', 's_f', 0.1)
    no_trans = parse_param(json_attr, 'no_trans', 's_s', 'False') == 'True'

    shadow_net.add_deformable_psroi_pooling(json_name, bottom_names, [json_name], output_dim, group_size, pooled_size, part_size, sample_per_part, spatial_scale, trans_std, no_trans)


def convert_eltwise(mxnet_nodes, index, param_dict, shadow_net, operation):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    shadow_net.add_eltwise(json_name, bottom_names, [json_name], operation)


def convert_leakyrelu(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)
    if len(param_names) > 0:
        param_dict[json_name] = param_names

    act_type = parse_param(json_attr, 'act_type', 's_s', 'leaky')
    slope = parse_param(json_attr, 'slope', 's_f', 0.25)
    lower_bound = parse_param(json_attr, 'lower_bound', 's_f', 0.125)
    upper_bound = parse_param(json_attr, 'upper_bound', 's_f', 0.334)
    if act_type == 'elu':
        raise ValueError('Unsupported activate type', act_type)
    elif act_type == 'leaky':
        act_type = 'Leaky'
    elif act_type == 'prelu':
        act_type = 'PRelu'
    elif act_type == 'rrelu':
        act_type = 'Leaky'
        slope = (lower_bound + upper_bound) / 2.0
    else:
        raise ValueError('Unsupported activate type', act_type)

    shadow_net.add_activate(json_name, bottom_names, [json_name], act_type, slope=slope)


def convert_pooling(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    pool = parse_param(json_attr, 'pool_type', 's_s', 'max')
    if pool == 'avg':
        pool = 'Ave'
    elif pool == 'max':
        pool = 'Max'
    elif pool == 'sum':
        raise ValueError('Currently not support pool type', pool)
    else:
        raise ValueError('Unsupported pool type', pool)
    kernel_size = parse_param(json_attr, 'kernel', 'v_i', [3, 3])
    stride = parse_param(json_attr, 'stride', 'v_i', [1, 1])
    pad = parse_param(json_attr, 'pad', 'v_i', [0, 0])
    global_pooling = parse_param(json_attr, 'global_pool', 's_s', 'False') == 'True'
    full_pooling = parse_param(json_attr, 'pooling_convention', 's_s', 'valid') == 'full'

    shadow_net.add_pooling(json_name, bottom_names, [json_name], pool, kernel_size[0], stride[0], pad[0], global_pooling, full_pooling)


def convert_proposal(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    feat_stride = parse_param(json_attr, 'feature_stride', 's_i', 16)
    pre_nms_top_n = parse_param(json_attr, 'rpn_pre_nms_top_n', 's_i', 6000)
    post_nms_top_n = parse_param(json_attr, 'rpn_post_nms_top_n', 's_i', 300)
    min_size = parse_param(json_attr, 'rpn_min_size', 's_i', 16)
    nms_thresh = parse_param(json_attr, 'threshold', 's_f', 0.7)
    ratios = parse_param(json_attr, 'ratios', 'v_f', [0.5, 1, 2])
    scales = parse_param(json_attr, 'scales', 'v_f', [8, 16, 32])

    shadow_net.add_proposal(json_name, bottom_names, [json_name], feat_stride, pre_nms_top_n, post_nms_top_n, min_size, nms_thresh, ratios, scales)


def convert_reshape(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    shape = parse_param(json_attr, 'shape', 'v_f', None)

    shadow_net.add_reshape(json_name, bottom_names, [json_name], shape)


def convert_roi_pooling(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    pooled_size = parse_param(json_attr, 'pooled_size', 'v_i', [14, 14])
    spatial_scale = parse_param(json_attr, 'spatial_scale', 's_f', 0.0625)

    shadow_net.add_roi_pooling(json_name, bottom_names, [json_name], pooled_size[0], pooled_size[1], spatial_scale)


def convert_softmax(mxnet_nodes, index, param_dict, shadow_net):
    json_node = mxnet_nodes[index]
    json_name = json_node['name']
    json_inputs = json_node['inputs']
    if params_str in json_node:
        json_attr = json_node[params_str]
    else:
        json_attr = {}
    bottom_names, param_names = find_inputs(mxnet_nodes, json_name, json_inputs)

    shadow_net.add_softmax(json_name, bottom_names, [json_name])


def mxnet2shadow(model_root, meta_net_info, copy_params=False):
    shadow_net = Shadow(meta_net_info['save_name'])

    for n, model_name in enumerate(meta_net_info['model_name']):
        model_epoch = meta_net_info['model_epoch'][n]
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_root + '/' + model_name, model_epoch)
        mxnet_symbol = json.loads(sym.tojson())
        mxnet_nodes = mxnet_symbol['nodes']
        mxnet_heads = mxnet_symbol['heads']

        net_info = meta_net_info['network'][n]
        net_info['arg']['out_blob_v_s'] = find_nodes(mxnet_nodes, mxnet_heads)

        shadow_net.set_net(n)
        shadow_net.set_net_name(meta_net_info['model_name'][n])
        shadow_net.set_net_arg(net_info['arg'])

        convert_input(net_info, shadow_net)

        param_dict = {}
        for index, json_node in enumerate(mxnet_nodes):
            json_op = json_node['op']
            if json_op == 'null':
                continue
            elif json_op == 'Activation':
                convert_activate(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'BatchNorm':
                convert_batch_norm(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == '_plus_scalar' or json_op == '_minus_scalar' or json_op == '_mul_scalar' or json_op == '_div_scalar':
                convert_binary(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'Concat':
                convert_concat(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'FullyConnected':
                convert_connected(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'Convolution':
                convert_conv(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == '_contrib_DeformableConvolution':
                convert_deformable_conv(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == '_contrib_DeformablePSROIPooling':
                convert_deformable_psroi_pooling(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'elemwise_add' or json_op == 'broadcast_add':
                convert_eltwise(mxnet_nodes, index, param_dict, shadow_net, 'Sum')
            elif json_op == 'LeakyReLU':
                convert_leakyrelu(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'Pooling':
                convert_pooling(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == '_contrib_Proposal':
                convert_proposal(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'Reshape':
                convert_reshape(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'ROIPooling':
                convert_roi_pooling(mxnet_nodes, index, param_dict, shadow_net)
            elif json_op == 'SoftmaxOutput' or json_op == 'SoftmaxActivation' or json_op == 'softmax':
                convert_softmax(mxnet_nodes, index, param_dict, shadow_net)
            else:
                print('Skipping ' + json_op, ' please check!')

        if copy_params:
            copy_weights(arg_params, aux_params, param_dict, shadow_net)

    return shadow_net
