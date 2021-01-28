import copy
from google.protobuf import text_format
from proto import caffe_pb2


def copy_weights(caffe_model, network):
    net_param = network.net_param
    for op in net_param.op:
        op_name = op.name
        for caffe_layer in caffe_model.layer:
            if caffe_layer.name == op_name:
                for n, caffe_blob in enumerate(caffe_layer.blobs):
                    blob_shape = caffe_blob.shape.dim
                    if len(blob_shape) == 0:
                        blob_shape = [len(caffe_blob.data)]
                    blob = net_param.blob.add()
                    blob.name = op_name + '/weights:{}'.format(n)
                    blob.shape.extend(blob_shape)
                    blob.data_f.extend(caffe_blob.data)
                    op.bottom.append(blob.name)
                break


def convert_input(caffe_deploy, net_info, network):
    start_layer = 0
    inputs, shapes = [], []

    if len(caffe_deploy.input) > 0:
        inputs.extend(caffe_deploy.input)
        if len(net_info['input_shape']) == 0:
            if len(caffe_deploy.input_shape) > 0:
                for caffe_shape in caffe_deploy.input_shape:
                    shapes.append(caffe_shape.dim)
            elif len(caffe_deploy.input_dim) > 0:
                num_dim = len(caffe_deploy.input_dim)
                num_shape = int(num_dim / 4)
                for i in range(0, num_shape):
                    shapes.append(caffe_deploy.input_dim[4*i : 4*(i+1)])
        else:
            shapes = net_info['input_shape']
        start_layer = 0
    elif caffe_deploy.layer[0].type == 'Input':
        caffe_input_layer = caffe_deploy.layer[0]
        inputs.extend(caffe_input_layer.top)
        if len(net_info['input_shape']) == 0:
            if caffe_input_layer.HasField('input_param'):
                caffe_param = caffe_input_layer.input_param
                for caffe_shape in caffe_param.shape:
                    shapes.append(caffe_shape.dim)
        else:
            shapes = net_info['input_shape']
        start_layer = 1

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
            if 'data' in input_name:
                network.add_scale(input_name, [input_name], [input_name], 1, False, False, scale_value, mean_value)

    return start_layer


def convert_activate(caffe_layer, network):
    layer_name = caffe_layer.name
    layer_type = caffe_layer.type
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    if layer_type == 'PReLU':
        act_type = 'PRelu'
    elif layer_type == 'ReLU':
        act_type = 'Relu'
    elif layer_type == 'Sigmoid':
        act_type = 'Sigmoid'
    else:
        raise ValueError('Unsupported activate type', layer_type)

    network.add_activate(layer_name, bottom_names, top_names, act_type)


def convert_axpy(caffe_layer, network):
    layer_name = caffe_layer.name
    layer_type = caffe_layer.type
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    network.add_axpy(layer_name, bottom_names, top_names)


def convert_batch_norm(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    use_global_stats = True
    eps = 1e-5
    if caffe_layer.HasField('batch_norm_param'):
        caffe_param = caffe_layer.batch_norm_param
        if caffe_param.HasField('use_global_stats'):
            use_global_stats = caffe_param.use_global_stats
        if caffe_param.HasField('eps'):
            eps = caffe_param.eps

    network.add_batch_norm(layer_name, bottom_names, top_names, use_global_stats, eps)


def convert_bias(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    axis = 1
    num_axes = 1
    if caffe_layer.HasField('bias_param'):
        caffe_param = caffe_layer.bias_param
        if caffe_param.HasField('axis'):
            axis = caffe_param.axis
        if caffe_param.HasField('num_axes'):
            num_axes = caffe_param.num_axes

    network.add_scale(layer_name, bottom_names, top_names, axis, False, True)


def convert_concat(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    axis = 1
    if caffe_layer.HasField('concat_param'):
        caffe_param = caffe_layer.concat_param
        if caffe_param.HasField('axis'):
            axis = caffe_param.axis

    network.add_concat(layer_name, bottom_names, top_names, axis)


def convert_connected(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    bias_term = True
    transpose = True
    if caffe_layer.HasField('inner_product_param'):
        caffe_param = caffe_layer.inner_product_param
        if caffe_param.HasField('num_output'):
            num_output = caffe_param.num_output
        else:
            raise ValueError('num_output must be supplied')
        if caffe_param.HasField('bias_term'):
            bias_term = caffe_param.bias_term
        if caffe_param.HasField('transpose'):
            transpose = not caffe_param.transpose

    network.add_connected(layer_name, bottom_names, top_names, num_output, bias_term, transpose)


def convert_conv(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    stride = 1
    pad = 0
    dilation = 1
    group = 1
    bias_term = True
    if caffe_layer.HasField('convolution_param'):
        caffe_param = caffe_layer.convolution_param
        if caffe_param.HasField('num_output'):
            num_output = caffe_param.num_output
        else:
            raise ValueError('num_output must be supplied')
        if len(caffe_param.kernel_size) > 0:
            kernel_size = caffe_param.kernel_size
        else:
            raise ValueError('kernel_size must be supplied')
        if len(caffe_param.stride) > 0:
            stride = caffe_param.stride
        if len(caffe_param.pad) > 0:
            pad = caffe_param.pad
        if len(caffe_param.dilation) > 0:
            dilation = caffe_param.dilation[0]
        if caffe_param.HasField('group'):
            group = caffe_param.group
        if caffe_param.HasField('bias_term'):
            bias_term = caffe_param.bias_term

    network.add_conv(layer_name, bottom_names, top_names, num_output, kernel_size, stride, pad, dilation, group, bias_term)


def convert_decode_box(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    method = 0 if len(bottom_names) == 3 else 1
    num_classes = 81
    background_label_id = 0
    objectness_score = 0.01
    if caffe_layer.HasField('detection_output_param'):
        caffe_param = caffe_layer.detection_output_param
        if caffe_param.HasField('num_classes'):
            num_classes = caffe_param.num_classes
        if caffe_param.HasField('background_label_id'):
            background_label_id = caffe_param.background_label_id
        if caffe_param.HasField('objectness_score'):
            objectness_score = caffe_param.objectness_score
        if caffe_param.HasField('code_type'):
            assert caffe_param.code_type == 2
        if caffe_param.HasField('variance_encoded_in_target'):
            assert caffe_param.variance_encoded_in_target is False
        if caffe_param.HasField('share_location'):
            assert caffe_param.share_location is True

    network.add_decode_box(layer_name, bottom_names, top_names, method, num_classes, True, background_label_id, objectness_score)


def convert_deconv(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    stride = 1
    pad = 0
    dilation = 1
    group = 1
    bias_term = True
    if caffe_layer.HasField('convolution_param'):
        caffe_param = caffe_layer.convolution_param
        if caffe_param.HasField('num_output'):
            num_output = caffe_param.num_output
        else:
            raise ValueError('num_output must be supplied')
        if len(caffe_param.kernel_size) > 0:
            kernel_size = caffe_param.kernel_size
        else:
            raise ValueError('kernel_size must be supplied')
        if len(caffe_param.stride) > 0:
            stride = caffe_param.stride
        if len(caffe_param.pad) > 0:
            pad = caffe_param.pad
        if len(caffe_param.dilation) > 0:
            dilation = caffe_param.dilation[0]
        if caffe_param.HasField('group'):
            group = caffe_param.group
        if caffe_param.HasField('bias_term'):
            bias_term = caffe_param.bias_term

    network.add_deconv(layer_name, bottom_names, top_names, num_output, kernel_size, stride, pad, dilation, group, bias_term)


def convert_eltwise(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    eltwise_op = 'Sum'
    coeff = None
    if caffe_layer.HasField('eltwise_param'):
        caffe_param = caffe_layer.eltwise_param
        if caffe_param.HasField('operation'):
            if caffe_param.operation == 0:
                eltwise_op = 'Prod'
            elif caffe_param.operation == 1:
                eltwise_op = 'Sum'
            elif caffe_param.operation == 2:
                eltwise_op = 'Max'
        if len(caffe_param.coeff) > 0:
            coeff = caffe_param.coeff

    network.add_eltwise(layer_name, bottom_names, top_names, eltwise_op, coeff)


def convert_flatten(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    axis = 1
    end_axis = -1
    if caffe_layer.HasField('flatten_param'):
        caffe_param = caffe_layer.flatten_param
        if caffe_param.HasField('axis'):
            axis = caffe_param.axis
        if caffe_param.HasField('end_axis'):
            end_axis = caffe_param.end_axis

    network.add_flatten(layer_name, bottom_names, top_names, axis, end_axis)


def convert_lrn(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    local_size = 5
    alpha = 1
    beta = 0.75
    norm_region = 'AcrossChannels'
    k = 1
    if caffe_layer.HasField('lrn_param'):
        caffe_param = caffe_layer.lrn_param
        if caffe_param.HasField('local_size'):
            local_size = caffe_param.local_size
        if caffe_param.HasField('alpha'):
            alpha = caffe_param.alpha
        if caffe_param.HasField('beta'):
            beta = caffe_param.beta
        if caffe_param.HasField('norm_region'):
            if caffe_param.norm_region == 0:
                norm_region = 'AcrossChannels'
            elif caffe_param.norm_region == 1:
                norm_region = 'WithinChannel'
        if caffe_param.HasField('k'):
            k = caffe_param.k

    network.add_lrn(layer_name, bottom_names, top_names, local_size, alpha, beta, norm_region, k)


def convert_normalize(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    across_spatial = True
    channel_shared = True
    eps = 1e-5
    if caffe_layer.HasField('norm_param'):
        caffe_param = caffe_layer.norm_param
        if caffe_param.HasField('across_spatial'):
            across_spatial = caffe_param.across_spatial
        if caffe_param.HasField('channel_shared'):
            channel_shared = caffe_param.channel_shared
        if caffe_param.HasField('eps'):
            eps = caffe_param.eps

    network.add_ssd_normalize(layer_name, bottom_names, top_names, across_spatial, channel_shared, eps)


def convert_permute(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    order = None
    if caffe_layer.HasField('permute_param'):
        caffe_param = caffe_layer.permute_param
        if len(caffe_param.order) > 0:
            order = caffe_param.order

    network.add_permute(layer_name, bottom_names, top_names, order)


def convert_pooling(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    pool = 'Max'
    kernel_size = [3, 3]
    stride = [1, 1]
    pad = [0, 0]
    global_pooling = False
    if caffe_layer.HasField('pooling_param'):
        caffe_param = caffe_layer.pooling_param
        if caffe_param.HasField('global_pooling'):
            global_pooling = caffe_param.global_pooling
        if caffe_param.HasField('pool'):
            if caffe_param.pool == 0:
                pool = 'Max'
            elif caffe_param.pool == 1:
                pool = 'Ave'
        if caffe_param.HasField('kernel_size'):
            kernel_size = [caffe_param.kernel_size] * 2
        else:
            if not global_pooling:
                raise ValueError('kernel_size must be supplied')
        if caffe_param.HasField('stride'):
            stride = [caffe_param.stride] * 2
        if caffe_param.HasField('pad'):
            pad = [caffe_param.pad] * 2

    network.add_pooling(layer_name, bottom_names, top_names, pool, kernel_size, stride, pad, global_pooling)


def convert_prior_box(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    min_size = None
    max_size = None
    aspect_ratio = None
    flip = True
    clip = True
    variance = None
    step = 0
    offset = 0.5
    if caffe_layer.HasField('prior_box_param'):
        caffe_param = caffe_layer.prior_box_param
        if len(caffe_param.min_size) > 0:
            min_size = caffe_param.min_size
        if len(caffe_param.max_size) > 0:
            max_size = caffe_param.max_size
        if len(caffe_param.aspect_ratio) > 0:
            aspect_ratio = caffe_param.aspect_ratio
        if caffe_param.HasField('flip'):
            flip = caffe_param.flip
        if caffe_param.HasField('clip'):
            clip = caffe_param.clip
        if len(caffe_param.variance) > 0:
            variance = caffe_param.variance
        if caffe_param.HasField('step'):
            step = caffe_param.step
        if caffe_param.HasField('offset'):
            offset = caffe_param.offset

    network.add_prior_box(layer_name, bottom_names, top_names, min_size, max_size, aspect_ratio, flip, clip, variance, step, offset)


def convert_python(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    if caffe_layer.HasField('python_param'):
        caffe_param = caffe_layer.python_param
        if caffe_param.HasField('layer'):
            if caffe_param.layer == 'ProposalLayer':
                if caffe_param.HasField('param_str'):
                    print('Can not parse python param, please check ' + caffe_param.param_str)
                network.add_proposal(layer_name, bottom_names, top_names)
            else:
                raise ValueError('Layer not support ' + caffe_param.layer)
        else:
            raise ValueError('Must have layer field')
    else:
        raise ValueError('Must have python param')


def convert_psroi_pooling(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    if caffe_layer.HasField('psroi_pooling_param'):
        caffe_param = caffe_layer.psroi_pooling_param
        if caffe_param.HasField('group_size'):
            group_size = caffe_param.group_size
        else:
            raise ValueError('group_size must be supplied')
        if caffe_param.HasField('spatial_scale'):
            spatial_scale = caffe_param.spatial_scale
        else:
            raise ValueError('spatial_scale must be supplied')

    network.add_psroi_pooling(layer_name, bottom_names, top_names, group_size, group_size, spatial_scale)


def convert_reshape(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    shape = None
    axis = 0
    num_axes = -1
    if caffe_layer.HasField('reshape_param'):
        caffe_param = caffe_layer.reshape_param
        if len(caffe_param.shape.dim) > 0:
            shape = caffe_param.shape.dim
        if caffe_param.HasField('axis'):
            axis = caffe_param.axis
        if caffe_param.HasField('num_axes'):
            num_axes = caffe_param.num_axes

    network.add_reshape(layer_name, bottom_names, top_names, shape, axis, num_axes)


def convert_roi_pooling(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    if caffe_layer.HasField('roi_pooling_param'):
        caffe_param = caffe_layer.roi_pooling_param
        if caffe_param.HasField('pooled_h'):
            pooled_h = caffe_param.pooled_h
        else:
            raise ValueError('pooled_h must be supplied')
        if caffe_param.HasField('pooled_w'):
            pooled_w = caffe_param.pooled_w
        else:
            raise ValueError('pooled_w must be supplied')
        if caffe_param.HasField('spatial_scale'):
            spatial_scale = caffe_param.spatial_scale
        else:
            raise ValueError('spatial_scale must be supplied')

    network.add_roi_pooling(layer_name, bottom_names, top_names, pooled_h, pooled_w, spatial_scale)


def convert_scale(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    axis = 1
    num_axes = 1
    bias_term = False
    if caffe_layer.HasField('scale_param'):
        caffe_param = caffe_layer.scale_param
        if caffe_param.HasField('axis'):
            axis = caffe_param.axis
        if caffe_param.HasField('num_axes'):
            num_axes = caffe_param.num_axes
        if caffe_param.HasField('bias_term'):
            bias_term = caffe_param.bias_term

    network.add_scale(layer_name, bottom_names, top_names, axis, True, bias_term)


def convert_slice(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    axis = 1
    slice_point = None
    if caffe_layer.HasField('slice_param'):
        caffe_param = caffe_layer.slice_param
        if caffe_param.HasField('axis'):
            axis = caffe_param.axis
        if len(caffe_param.slice_point) > 0:
            slice_point = caffe_param.slice_point

    network.add_slice(layer_name, bottom_names, top_names, axis, slice_point)


def convert_softmax(caffe_layer, network):
    layer_name = caffe_layer.name
    bottom_names = caffe_layer.bottom
    top_names = caffe_layer.top

    axis = 1
    if caffe_layer.HasField('softmax_param'):
        caffe_param = caffe_layer.softmax_param
        if caffe_param.HasField('axis'):
            axis = caffe_param.axis

    network.add_softmax(layer_name, bottom_names, top_names, axis)


def convert_caffe(network, net_info, model_root, model_name, copy_params):
    deploy_file = model_root + '/' + model_name + '.prototxt'
    deploy_model = model_root + '/' + model_name + '.caffemodel'

    caffe_deploy = caffe_pb2.NetParameter()
    with open(deploy_file, 'r') as caffe_file:
        text_format.Merge(caffe_file.read(), caffe_deploy)

    network.set_net_name(model_name)
    network.set_net_arg_dict(net_info['arg'])

    start_layer = convert_input(caffe_deploy, net_info, network)

    for index in range(start_layer, len(caffe_deploy.layer)):
        caffe_layer = caffe_deploy.layer[index]
        layer_type = caffe_layer.type
        if layer_type == 'PReLU' or layer_type == 'ReLU' or layer_type == 'Sigmoid':
            convert_activate(caffe_layer, network)
        elif layer_type == 'Axpy':
            convert_axpy(caffe_layer, network)
        elif layer_type == 'BatchNorm':
            convert_batch_norm(caffe_layer, network)
        elif layer_type == 'Bias':
            convert_bias(caffe_layer, network)
        elif layer_type == 'Concat':
            convert_concat(caffe_layer, network)
        elif layer_type == 'InnerProduct':
            convert_connected(caffe_layer, network)
        elif layer_type == 'Convolution' or layer_type == 'DepthwiseConvolution':
            convert_conv(caffe_layer, network)
        elif layer_type == 'DetectionOutput':
            convert_decode_box(caffe_layer, network)
        elif layer_type == 'Deconvolution':
            convert_deconv(caffe_layer, network)
        elif layer_type == 'Eltwise':
            convert_eltwise(caffe_layer, network)
        elif layer_type == 'Flatten':
            convert_flatten(caffe_layer, network)
        elif layer_type == 'LRN':
            convert_lrn(caffe_layer, network)
        elif layer_type == 'Normalize':
            convert_normalize(caffe_layer, network)
        elif layer_type == 'Permute':
            convert_permute(caffe_layer, network)
        elif layer_type == 'Pooling':
            convert_pooling(caffe_layer, network)
        elif layer_type == 'PriorBox':
            convert_prior_box(caffe_layer, network)
        elif layer_type == 'PSROIPooling':
            convert_psroi_pooling(caffe_layer, network)
        elif layer_type == 'Python':
            convert_python(caffe_layer, network)
        elif layer_type == 'Reshape':
            convert_reshape(caffe_layer, network)
        elif layer_type == 'ROIPooling':
            convert_roi_pooling(caffe_layer, network)
        elif layer_type == 'Scale':
            convert_scale(caffe_layer, network)
        elif layer_type == 'Slice':
            convert_slice(caffe_layer, network)
        elif layer_type == 'Softmax':
            convert_softmax(caffe_layer, network)
        else:
            print('Layer type: ' + layer_type + ' is not recognized!')

    if copy_params:
        caffe_model = caffe_pb2.NetParameter()
        with open(deploy_model, 'rb') as caffe_file:
            caffe_model.ParseFromString(caffe_file.read())
        copy_weights(caffe_model, network)
