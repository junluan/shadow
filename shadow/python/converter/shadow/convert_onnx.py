import copy
import numpy as np
import onnx


onnx_ver = [int(v) for v in onnx.__version__.split('.')]
onnx_ver_int = 10000 * onnx_ver[0] + 100 * onnx_ver[1] + onnx_ver[2]
assert onnx_ver_int >= 10300


def optimize_onnx(onnx_model):
    passes = ['nop', 'extract_constant_to_initializer',
              'eliminate_identity', 'eliminate_nop_pad',
              'eliminate_nop_transpose', 'eliminate_unused_initializer',
              'fuse_add_bias_into_conv', 'fuse_bn_into_conv',
              'fuse_consecutive_squeezes', 'fuse_consecutive_transposes',
              'fuse_transpose_into_gemm']
    if onnx_ver_int >= 10400:
        passes += ['eliminate_nop_dropout', 'fuse_consecutive_concats',
                   'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv']
    from onnx import optimizer
    all_passes = optimizer.get_available_passes()
    for p in passes:
        assert p in all_passes
    return optimizer.optimize(onnx_model, passes)


def get_param_shape(onnx_initializer, name):
    for initializer in onnx_initializer:
        if initializer.name == name:
            return initializer.dims
    raise ValueError(name, 'not in initializer blobs')


def get_param_weight(onnx_initializer, name):
    for initializer in onnx_initializer:
        if initializer.name == name:
            if len(initializer.float_data) > 0:
                param_data = np.asarray(initializer.float_data, dtype=np.float32)
            elif len(initializer.int32_data) > 0:
                param_data = np.asarray(initializer.int32_data, dtype=np.int32)
            elif len(initializer.int64_data) > 0:
                param_data = np.asarray(initializer.int64_data, dtype=np.int64)
            else:
                assert initializer.HasField('raw_data')
                if initializer.data_type == onnx.TensorProto.FLOAT:
                    data_type = np.float32
                elif initializer.data_type == onnx.TensorProto.DOUBLE:
                    data_type = np.double
                elif initializer.data_type == onnx.TensorProto.INT32:
                    data_type = np.int32
                elif initializer.data_type == onnx.TensorProto.INT64:
                    data_type = np.int64
                else:
                    raise ValueError(initializer.data_type, 'not supported')
                param_data = np.frombuffer(initializer.raw_data, dtype=data_type)
            return param_data, initializer.dims
    raise ValueError(name, 'not in initializer blobs')


def copy_weights(onnx_initializer, param_dict, network):
    net_param = network.net_param
    for op in net_param.op:
        op_name = op.name
        if op_name not in param_dict:
            continue
        for n, param_name in enumerate(param_dict[op_name]):
            param_data, param_shape = get_param_weight(onnx_initializer, param_name)
            blob = net_param.blob.add()
            blob.name = op_name + '/weights:{}'.format(n)
            blob.shape.extend(param_shape)
            if param_data.dtype == np.float32:
                blob.type = 'float'
                blob.data_f.extend(param_data)
            elif param_data.dtype == np.int32 or param_data.dtype == np.int64:
                blob.type = 'int'
                blob.data_i.extend(param_data.astype(np.int32))
            else:
                raise ValueError('Unsupported data type', param_data.dtype)
            op.bottom.append(blob.name)


def parse_value_info(blob):
    name = blob.name
    tensor_type = blob.type.tensor_type
    elem_type = tensor_type.elem_type
    shape = [int(dim.dim_value) for dim in tensor_type.shape.dim]
    return {'name': name, 'data_type': elem_type, 'shape': shape}


def parse_graph(onnx_graph):
    onnx_input = onnx_graph.input
    onnx_output = onnx_graph.output
    onnx_initializer = onnx_graph.initializer

    input_infos, output_names = [], []
    for input in onnx_input:
        name = input.name
        is_weight = False
        for initializer in onnx_initializer:
            if name == initializer.name:
                is_weight = True
                break
        if not is_weight:
            input_infos.append(parse_value_info(input))
    for output in onnx_output:
        output_names.append(output.name)

    return input_infos, output_names


def parse_attribute(attributes, name, default_value):
    attribute = None
    for attr in attributes:
        if name == attr.name:
            attribute = attr
            break
    if attribute is not None:
        attribute_type = attribute.type
        if attribute_type == onnx.AttributeProto.INT:
            return int(attribute.i)
        elif attribute_type == onnx.AttributeProto.FLOAT:
            return float(attribute.f)
        elif attribute_type == onnx.AttributeProto.STRING:
            return str(attribute.s, encoding='utf-8')
        elif attribute_type == onnx.AttributeProto.INTS:
            return [int(p) for p in attribute.ints]
        elif attribute_type == onnx.AttributeProto.FLOATS:
            return [float(p) for p in attribute.floats]
        elif attribute_type == onnx.AttributeProto.STRINGS:
            return [str(p, encoding='utf-8') for p in attribute.strings]
    else:
        return default_value


def find_inputs(onnx_nodes, index, onnx_inputs):
    def find_input_node(onnx_nodes, input_name, index):
        for i in range(0, index):
            node = onnx_nodes[index - 1 - i]
            if input_name in node.output:
                return node
        return None

    bottom_names = []
    for input_name in onnx_inputs:
        bottom_node = find_input_node(onnx_nodes, input_name, index)
        if bottom_node is not None:
            if bottom_node.op_type == 'Dropout':
                inner_bottom_names = find_inputs(onnx_nodes, index - 1, bottom_node.input)
                bottom_names.extend(inner_bottom_names)
            else:
                bottom_names.append(input_name)
    if len(bottom_names) == 0:
        bottom_names.extend(onnx_inputs)
    return bottom_names


def check_inputs(onnx_initializer, onnx_inputs):
    bottoms, params = [], []
    for input_name in onnx_inputs:
        no_constant = True
        for initializer in onnx_initializer:
            if input_name == initializer.name:
                no_constant = False
                break
        if no_constant:
            bottoms.append(input_name)
        else:
            params.append(input_name)
    return bottoms, params


def convert_input(net_info, network, input_infos):
    inputs, shapes = [], []
    for input in input_infos:
        inputs.append(input['name'])
        shapes.append(input['shape'])

    assert len(inputs) == len(shapes)
    network.add_input('input', [], inputs, shapes)

    num_mean = len(net_info['mean_value'])
    num_scale = len(net_info['scale_value'])
    mean_value = copy.deepcopy(net_info['mean_value']) if num_mean > 0 else None
    scale_value = copy.deepcopy(net_info['scale_value']) if num_scale > 0 else None
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
        for input_name in inputs:
            network.add_scale(input_name, [input_name], [input_name], 1, False, False, scale_value, mean_value)


def convert_activate(onnx_nodes, index, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_type = onnx_node.op_type
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    if op_type == 'PRelu':
        act_type = 'PRelu'
        bottom_names = find_inputs(onnx_nodes, index, op_input[:1])
        param_dict[op_name] = op_input[1:]
    elif op_type == 'Relu':
        act_type = 'Relu'
    elif op_type == 'Sigmoid':
        act_type = 'Sigmoid'
    else:
        raise ValueError('Unsupported activate type', op_type)

    network.add_activate(op_name, bottom_names, top_names, act_type)


def convert_batch_norm(onnx_nodes, index, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input[:1])
    top_names = op_output[:1]
    param_dict[op_name] = op_input[3:5]
    param_dict[op_name + '_scale'] = op_input[1:3]

    use_global_stats = True
    eps = parse_attribute(op_attribute, 'epsilon', 1e-5)

    network.add_batch_norm(op_name, bottom_names, top_names, use_global_stats, eps)
    network.add_scale(op_name + '_scale', top_names, top_names, 1)


def convert_binary(onnx_nodes, index, onnx_initializer, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_type = onnx_node.op_type
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottoms, params = check_inputs(onnx_initializer, op_input)

    bottom_names = find_inputs(onnx_nodes, index, bottoms)
    top_names = op_output

    scalar = None
    if len(params) == 1:
        scalar_data, scalar_shape = get_param_weight(onnx_initializer, params[0])
        if len(scalar_data) == 1:
            scalar = scalar_data[0]
        else:
            param_dict[op_name] = params
    elif len(params) > 1:
        raise ValueError('Too many params', params)

    if op_type == 'Add':
        operation_type = 'Add'
    elif op_type == 'Sub':
        operation_type = 'Sub'
    elif op_type == 'Mul':
        operation_type = 'Mul'
    elif op_type == 'Div':
        operation_type = 'Div'
    else:
        raise ValueError('Unsupported binary type', op_type)

    network.add_binary(op_name, bottom_names, top_names, operation_type, scalar)


def convert_concat(onnx_nodes, index, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    axis = parse_attribute(op_attribute, 'axis', 1)

    network.add_concat(op_name, bottom_names, top_names, axis)


def convert_connected(onnx_nodes, index, onnx_initializer, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input[:1])
    top_names = op_output
    param_dict[op_name] = op_input[1:]

    alpha = parse_attribute(op_attribute, 'alpha', 1)
    beta = parse_attribute(op_attribute, 'beta', 1)
    transA = parse_attribute(op_attribute, 'transA', 0)
    transB = parse_attribute(op_attribute, 'transB', 0)

    weight_shape = get_param_shape(onnx_initializer, op_input[1])
    bias_shape = get_param_shape(onnx_initializer, op_input[2])
    num_output = weight_shape[0] if transB else weight_shape[1]
    bias_term = True

    assert transA == 0
    assert abs(alpha - 1) < 0.001
    assert abs(beta - 1) < 0.001
    assert len(bias_shape) == 1
    assert num_output == bias_shape[0]

    network.add_connected(op_name, bottom_names, top_names, num_output, bias_term, transB)


def convert_conv(onnx_nodes, index, onnx_initializer, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input[:1])
    top_names = op_output
    param_dict[op_name] = op_input[1:]

    kernel_size = parse_attribute(op_attribute, 'kernel_shape', [3, 3])
    stride = parse_attribute(op_attribute, 'strides', [1, 1])
    pad = parse_attribute(op_attribute, 'pads', [0, 0, 0, 0])
    dilate = parse_attribute(op_attribute, 'dilations', [1, 1])
    group = parse_attribute(op_attribute, 'group', 1)

    weight_shape = get_param_shape(onnx_initializer, op_input[1])
    num_output = weight_shape[0]
    bias_term = len(op_input) > 2

    assert kernel_size[0] == weight_shape[2]
    assert kernel_size[1] == weight_shape[3]

    assert pad[0] == pad[2]
    assert pad[1] == pad[3]

    network.add_conv(op_name, bottom_names, top_names, num_output, kernel_size, stride, pad[:2], dilate[0], group, bias_term)


def convert_deconv(onnx_nodes, index, onnx_initializer, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input[:1])
    top_names = op_output
    param_dict[op_name] = op_input[1:]

    kernel_size = parse_attribute(op_attribute, 'kernel_shape', [3, 3])
    stride = parse_attribute(op_attribute, 'strides', [1, 1])
    pad = parse_attribute(op_attribute, 'pads', [0, 0, 0, 0])
    dilate = parse_attribute(op_attribute, 'dilations', [1, 1])
    group = parse_attribute(op_attribute, 'group', 1)
    output_padding = parse_attribute(op_attribute, 'output_padding', None)
    output_shape = parse_attribute(op_attribute, 'output_shape', None)

    assert output_padding is None and output_shape is None

    weight_shape = get_param_shape(onnx_initializer, op_input[1])
    num_output = weight_shape[1]
    bias_term = len(op_input) > 2

    assert kernel_size[0] == weight_shape[2]
    assert kernel_size[1] == weight_shape[3]

    assert pad[0] == pad[2]
    assert pad[1] == pad[3]

    network.add_deconv(op_name, bottom_names, top_names, num_output, kernel_size, stride, pad[:2], dilate[0], group, bias_term)


def convert_flatten(onnx_nodes, index, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    axis = parse_attribute(op_attribute, 'axis', 1)
    assert axis == 1

    network.add_flatten(op_name, bottom_names, top_names, axis)


def convert_gather(onnx_nodes, index, onnx_initializer, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottoms, params = check_inputs(onnx_initializer, op_input)

    bottom_names = find_inputs(onnx_nodes, index, bottoms)
    top_names = op_output
    param_dict[op_name] = params

    axis = parse_attribute(op_attribute, 'axis', 0)

    network.add_gather(op_name, bottom_names, top_names, axis)


def convert_matmul(onnx_nodes, index, onnx_initializer, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottoms, params = check_inputs(onnx_initializer, op_input)

    bottom_names = find_inputs(onnx_nodes, index, bottoms)
    top_names = op_output
    param_dict[op_name] = params

    network.add_matmul(op_name, bottom_names, top_names)


def convert_pad(onnx_nodes, index, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    mode = parse_attribute(op_attribute, 'mode', 'constant')
    pads = parse_attribute(op_attribute, 'pads', None)
    value = parse_attribute(op_attribute, 'value', 0)

    assert 'constant' in mode
    assert len(pads) == 8
    pads = [pads[2], pads[6], pads[3], pads[7]]

    network.add_pad(op_name, bottom_names, top_names, pads, value)


def convert_permute(onnx_nodes, index, onnx_initializer, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottoms, params = check_inputs(onnx_initializer, op_input)

    bottom_names = find_inputs(onnx_nodes, index, bottoms)
    top_names = op_output
    param_dict[op_name] = params

    perm = parse_attribute(op_attribute, 'perm', None)

    network.add_permute(op_name, bottom_names, top_names, perm)


def convert_pooling(onnx_nodes, index, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_type = onnx_node.op_type
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    global_pooling = False
    if op_type == 'MaxPool':
        pool = 'Max'
    elif op_type == 'AveragePool':
        pool = 'Ave'
    elif op_type == 'GlobalMaxPool':
        pool = 'Max'
        global_pooling = True
    elif op_type == 'GlobalAveragePool':
        pool = 'Ave'
        global_pooling = True
    else:
        raise ValueError('Unsupported pool type', op_type)
    kernel_size = parse_attribute(op_attribute, 'kernel_shape', [3, 3])
    stride = parse_attribute(op_attribute, 'strides', [1, 1])
    pad = parse_attribute(op_attribute, 'pads', [0, 0, 0, 0])
    full_pooling = False

    network.add_pooling(op_name, bottom_names, top_names, pool, kernel_size, stride, pad[0:2], global_pooling, full_pooling)


def convert_reduce(onnx_nodes, index, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_type = onnx_node.op_type
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    if op_type == 'ReduceProd':
        operation = 'Prod'
    elif op_type == 'ReduceSum':
        operation = 'Sum'
    elif op_type == 'ReduceMax':
        operation = 'Max'
    elif op_type == 'ReduceMin':
        operation = 'Min'
    elif op_type == 'ReduceMean':
        operation = 'Avg'
    else:
        raise ValueError('Unsupported reduce type', op_type)

    axes = parse_attribute(op_attribute, 'axes', None)
    keepdims = parse_attribute(op_attribute, 'keepdims', True)

    network.add_reduce(op_name, bottom_names, top_names, operation, axes, keepdims)


def convert_reshape(onnx_nodes, index, onnx_initializer, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input[:1])
    top_names = op_output

    shape_data, _ = get_param_weight(onnx_initializer, op_input[1])

    network.add_reshape(op_name, bottom_names, top_names, shape=shape_data)


def convert_resize(onnx_nodes, index, onnx_initializer, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input[:1])
    top_names = op_output

    mode = parse_attribute(op_attribute, 'mode', 'nearest')
    if mode == 'linear':
        mode = 'bilinear'

    scale_data, _ = get_param_weight(onnx_initializer, op_input[1])
    assert len(scale_data) == 4

    network.add_resize(op_name, bottom_names, top_names, scale=scale_data[2:], resize_type=mode)


def convert_softmax(onnx_nodes, index, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    axis = parse_attribute(op_attribute, 'axis', 1)

    network.add_softmax(op_name, bottom_names, top_names, axis)


def convert_squeeze(onnx_nodes, index, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    axes = parse_attribute(op_attribute, 'axes', None)

    network.add_squeeze(op_name, bottom_names, top_names, axes)


def convert_unsqueeze(onnx_nodes, index, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    axes = parse_attribute(op_attribute, 'axes', None)

    network.add_unsqueeze(op_name, bottom_names, top_names, axes)


def convert_aten(onnx_nodes, index, onnx_initializer, param_dict, network):
    onnx_node = onnx_nodes[index]
    op_input = onnx_node.input
    op_output = onnx_node.output
    op_attribute = onnx_node.attribute
    op_name = onnx_node.name if onnx_node.HasField('name') else 'node_{}'.format(index)

    bottom_names = find_inputs(onnx_nodes, index, op_input)
    top_names = op_output

    operator = parse_attribute(op_attribute, 'operator', None)

    if operator == 'max':
        dim = parse_attribute(op_attribute, 'dim', None)
        dim = [dim] if dim is not None else None
        keepdim = parse_attribute(op_attribute, 'keepdim', True)
        network.add_reduce(op_name, bottom_names, top_names[:1], 'Max', dim, keepdim)
    else:
        raise ValueError('Unsupported aten operator', operator)


def convert_onnx(network, net_info, model_root, model_name, copy_params):
    model_file = model_root + '/' + model_name + '.onnx'

    onnx_model = optimize_onnx(onnx.load(model_file))
    onnx_graph = onnx_model.graph
    onnx_nodes = onnx_graph.node
    onnx_initializer = onnx_graph.initializer

    input_infos, output_names = parse_graph(onnx_graph)

    net_info['arg']['out_blob_v_s'] = output_names

    network.set_net_name(model_name)
    network.set_net_arg_dict(net_info['arg'])

    convert_input(net_info, network, input_infos)

    param_dict = {}
    for index, node in enumerate(onnx_nodes):
        op_type = node.op_type
        if op_type == 'Relu' or op_type == 'PRelu' or op_type == 'Sigmoid':
            convert_activate(onnx_nodes, index, param_dict, network)
        elif op_type == 'BatchNormalization':
            convert_batch_norm(onnx_nodes, index, param_dict, network)
        elif op_type == 'Add' or op_type == 'Sub' or op_type == 'Mul' or op_type == 'Div':
            convert_binary(onnx_nodes, index, onnx_initializer, param_dict, network)
        elif op_type == 'Concat':
            convert_concat(onnx_nodes, index, network)
        elif op_type == 'Gemm':
            convert_connected(onnx_nodes, index, onnx_initializer, param_dict, network)
        elif op_type == 'Conv':
            convert_conv(onnx_nodes, index, onnx_initializer, param_dict, network)
        elif op_type == 'ConvTranspose':
            convert_deconv(onnx_nodes, index, onnx_initializer, param_dict, network)
        elif op_type == 'MatMul':
            convert_matmul(onnx_nodes, index, onnx_initializer, param_dict, network)
        elif op_type == 'Flatten':
            convert_flatten(onnx_nodes, index, network)
        elif op_type == 'Gather':
            convert_gather(onnx_nodes, index, onnx_initializer, param_dict, network)
        elif op_type == 'Pad':
            convert_pad(onnx_nodes, index, network)
        elif op_type == 'Transpose':
            convert_permute(onnx_nodes, index, onnx_initializer, param_dict, network)
        elif op_type == 'MaxPool' or op_type == 'AveragePool' or op_type == 'GlobalMaxPool' or op_type == 'GlobalAveragePool':
            convert_pooling(onnx_nodes, index, network)
        elif op_type == 'ReduceProd' or op_type == 'ReduceSum' or op_type == 'ReduceMax' or op_type == 'ReduceMin' or op_type == 'ReduceMean':
            convert_reduce(onnx_nodes, index, network)
        elif op_type == 'Reshape':
            convert_reshape(onnx_nodes, index, onnx_initializer, network)
        elif op_type == 'Upsample':
            convert_resize(onnx_nodes, index, onnx_initializer, network)
        elif op_type == 'Softmax':
            convert_softmax(onnx_nodes, index, network)
        elif op_type == 'Squeeze':
            convert_squeeze(onnx_nodes, index, network)
        elif op_type == 'Unsqueeze':
            convert_unsqueeze(onnx_nodes, index, network)
        elif op_type == 'ATen':
            convert_aten(onnx_nodes, index, onnx_initializer, param_dict, network)
        else:
            print('Skipping ' + op_type + ', please check!')

    if copy_params:
        copy_weights(onnx_initializer, param_dict, network)
