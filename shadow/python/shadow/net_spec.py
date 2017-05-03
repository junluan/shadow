import math

import shadow_pb2

from google.protobuf import text_format


class Shadow:

    def __init__(self, name, blob_shape=True):
        self.net_param = shadow_pb2.NetParameter()
        self.net_param.name = name
        self.blobs = {}
        self.blob_shape = blob_shape

    def add_common(self, layer_param, layer_name, layer_type, bottoms, tops):
        layer_param.name = layer_name
        layer_param.type = layer_type
        for bottom in bottoms:
            layer_param.bottom.append(bottom)
        for top in tops:
            layer_param.top.append(top)

    def add_activate(self, name, bottoms, tops, activate_type='Relu'):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Activate', bottoms, tops)

        if activate_type == 'Linear':
            layer_param.activate_param.type = shadow_pb2.ActivateParameter.Linear
        elif activate_type == 'Relu':
            layer_param.activate_param.type = shadow_pb2.ActivateParameter.Relu
        elif activate_type == 'Leaky':
            layer_param.activate_param.type = shadow_pb2.ActivateParameter.Leaky

        in_shape = self.blobs[bottoms[0]]['shape']
        self.blobs[tops[0]] = {'shape': in_shape}

    def add_batch_norm(self, name, bottoms, tops, use_global_stats=True):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'BatchNorm', bottoms, tops)

        if use_global_stats != True:
            layer_param.batch_norm_param.use_global_stats = use_global_stats

        in_shape = self.blobs[bottoms[0]]['shape']
        self.blobs[tops[0]] = {'shape': in_shape}

        if use_global_stats and self.blob_shape:
            mean_blob = layer_param.blobs.add()
            mean_blob.shape.dim.append(in_shape[1])
            variance_blob = layer_param.blobs.add()
            variance_blob.shape.dim.append(in_shape[1])
            factor_blob = layer_param.blobs.add()
            factor_blob.shape.dim.append(1)

    def add_bias(self, name, bottoms, tops, axis=1, num_axes=1):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Bias', bottoms, tops)

        if axis != 1:
            layer_param.bias_param.axis = axis
        if num_axes != 1:
            layer_param.bias_param.num_axes = num_axes

        in_shape = self.blobs[bottoms[0]]['shape']
        self.blobs[tops[0]] = {'shape': in_shape}

        if self.blob_shape:
            bias_blob = layer_param.blobs.add()
            bias_blob.shape.dim.append(in_shape[axis])

    def add_concat(self, name, bottoms, tops, axis=1):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Concat', bottoms, tops)

        if axis != 1:
            layer_param.concat_param.axis = axis

        in_shape = self.blobs[bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[axis] = 0
        for bottom in bottoms:
            in_shape = self.blobs[bottom]['shape']
            out_shape[axis] += in_shape[axis]
        self.blobs[tops[0]] = {'shape': out_shape}

    def add_connected(self, name, bottoms, tops, num_output, bias_term=True, transpose=False):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Connected', bottoms, tops)

        layer_param.connected_param.num_output = num_output
        if bias_term != True:
            layer_param.connected_param.bias_term = bias_term
        if transpose != False:
            layer_param.connected_param.transpose = transpose

        in_shape = self.blobs[bottoms[0]]['shape']
        out_shape = [in_shape[0], num_output]
        self.blobs[tops[0]] = {'shape': out_shape}

        bottom_num = in_shape[1] * in_shape[2] * in_shape[3]

        if self.blob_shape:
            weight_blob = layer_param.blobs.add()
            weight_blob.shape.dim.append(num_output)
            weight_blob.shape.dim.append(bottom_num)

            if bias_term:
                bias_blob = layer_param.blobs.add()
                bias_blob.shape.dim.append(num_output)

    def add_convolution(self, name, bottoms, tops, num_output, kernel_size, stride=1, pad=0, dilation=1, bias_term=True):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Convolution', bottoms, tops)

        layer_param.convolution_param.num_output = num_output
        layer_param.convolution_param.kernel_size = kernel_size
        layer_param.convolution_param.stride = stride
        layer_param.convolution_param.pad = pad
        if dilation != 1:
            layer_param.convolution_param.dilation = dilation
        if bias_term != True:
            layer_param.convolution_param.bias_term = bias_term

        def convolution_out_size(dim, ks, sd, pa, dila):
            kernel_extent = dila * (ks - 1) + 1
            return (dim + 2 * pa - kernel_extent) / sd + 1

        in_shape = self.blobs[bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[1] = num_output
        out_shape[2] = convolution_out_size(
            in_shape[2], kernel_size, stride, pad, dilation)
        out_shape[3] = convolution_out_size(
            in_shape[3], kernel_size, stride, pad, dilation)
        self.blobs[tops[0]] = {'shape': out_shape}

        if self.blob_shape:
            weight_blob = layer_param.blobs.add()
            weight_blob.shape.dim.append(num_output)
            weight_blob.shape.dim.append(in_shape[1])
            weight_blob.shape.dim.append(kernel_size)
            weight_blob.shape.dim.append(kernel_size)

            if bias_term == True:
                bias_blob = layer_param.blobs.add()
                bias_blob.shape.dim.append(num_output)

    def add_data(self, name, bottoms, tops, input_shape=[], scale=1, mean_value=[]):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Data', bottoms, tops)

        for dim in input_shape:
            layer_param.data_param.data_shape.dim.append(dim)

        if scale != 1:
            layer_param.data_param.scale = scale
        for mean in mean_value:
            layer_param.data_param.mean_value.append(mean)

        self.blobs[tops[0]] = {'shape': input_shape}

    def add_flatten(self, name, bottoms, tops, axis=1, end_axis=-1):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Flatten', bottoms, tops)

        if axis != 1:
            layer_param.flatten_param.axis = axis
        if end_axis != -1:
            layer_param.flatten_param.end_axis = end_axis

        in_shape = self.blobs['in_blob']['shape']
        out_shape = []
        num_axes = len(in_shape)
        if end_axis == -1:
            end_axis = num_axes - 1
        for i in range(0, axis):
            out_shape.append(in_shape[i])
        count = 1
        for i in range(axis, end_axis + 1):
            count *= in_shape[i]
        out_shape.append(count)
        for i in range(end_axis + 1, num_axes):
            out_shape.append(in_shape[i])
        self.blobs[tops[0]] = {'shape': out_shape}

    def add_normalize(self, name, bottoms, tops, across_spatial=True, channel_shared=True):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Normalize', bottoms, tops)

        if across_spatial != True:
            layer_param.normalize_param.across_spatial = across_spatial
        if channel_shared != True:
            layer_param.normalize_param.channel_shared = channel_shared

        in_shape = self.blobs['in_blob']['shape']
        self.blobs[tops[0]] = {'shape': in_shape}

        if self.blob_shape:
            scale_blob = layer_param.blobs.add()
            if channel_shared:
                scale_blob.shape.dim.append(1)
            else:
                scale_blob.shape.dim.append(in_blob[1])

    def add_permute(self, name, bottoms, tops, order):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Permute', bottoms, tops)

        for o in order:
            layer_param.permute_param.order.append(o)

        in_shape = self.blobs[bottoms[0]]['shape']
        out_shape = []
        for o in order:
            out_shape.append(in_shape[o])
        self.blobs[tops[0]] = {'shape': out_shape}

    def add_pooling(self, name, bottoms, tops, pool, kernel_size, stride=1, pad=0, global_pooling=False):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Pooling', bottoms, tops)

        if pool == 'Ave':
            layer_param.pooling_param.pool = shadow_pb2.PoolingParameter.Ave
        else:
            layer_param.pooling_param.pool = shadow_pb2.PoolingParameter.Max
        layer_param.pooling_param.kernel_size = kernel_size
        layer_param.pooling_param.stride = stride
        layer_param.pooling_param.pad = pad
        if global_pooling != False:
            layer_param.pooling_param.global_pooling = global_pooling

        def pooling_out_size(dim, ks, sd, pa):
            return int(math.ceil(float(dim + 2 * pa - ks) / sd)) + 1

        in_shape = self.blobs[bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[2] = pooling_out_size(in_shape[2], kernel_size, stride, pad)
        out_shape[3] = pooling_out_size(in_shape[3], kernel_size, stride, pad)
        if pad:
            if (out_shape[2] - 1) * stride >= in_shape[2] + pad:
                out_shape[2] = out_shape[2] - 1
            if (out_shape[3] - 1) * stride >= in_shape[3] + pad:
                out_shape[3] = out_shape[3] - 1
        self.blobs[tops[0]] = {'shape': out_shape}

    def add_prior_box(self, name, bottoms, tops, min_size, max_size, aspect_ratio=[], flip=True, clip=True, variance=[]):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'PriorBox', bottoms, tops)

        layer_param.prior_box_param.min_size = min_size
        layer_param.prior_box_param.max_size = max_size
        for asp in aspect_ratio:
            layer_param.prior_box_param.aspect_ratio.append(asp)
        if flip != True:
            layer_param.prior_box_param.flip = flip
        if clip != True:
            layer_param.prior_box_param.clip = clip
        for var in variance:
            layer_param.prior_box_param.variance.append(var)

        in_shape = self.blobs['in_blob']['shape']
        num_asp = len(aspect_ratio)
        if 1 in aspect_ratio:
            num_asp -= 1
        if flip:
            num_asp *= 2
        num_priors = 1 + num_asp
        out_shape = [1, 2, in_shape[2] * in_shape[3] * num_priors * 4]
        self.blobs[tops[0]] = {'shape': out_shape}

    def add_reorg(self, name, bottoms, tops, stride=2):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Reorg', bottoms, tops)

        layer_param.reorg_param.stride = stride

        in_shape = self.blobs[bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[1] = in_shape[1] * stride * stride
        out_shape[2] = in_shape[2] / stride
        out_shape[3] = in_shape[3] / stride
        self.blobs[tops[0]] = {'shape': out_shape}

    def add_reshape(self, name, bottoms, tops, shape, axis=0, num_axes=-1):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Reshape', bottoms, tops)

        for dim in shape:
            layer_param.reshape_param.shape.dim.append(dim)
        if axis != 0:
            layer_param.reshape_param.axis = axis
        if num_axes != -1:
            layer_param.reshape_param.num_axes = num_axes

        in_shape = self.blobs[bottoms[0]]['shape']
        out_shape = []
        inferred_axis = -1
        copy_axes = []
        constant_count = 1
        for i in range(0, len(shape)):
            dim = shape[i]
            if dim == 0:
                copy_axes.append(i)
            elif dim == -1:
                inferred_axis = i
            else:
                constant_count *= dim
        start_axis = axis
        if axis < 0:
            start_axis = len(in_shape) + axis + 1
        end_axis = len(in_shape)
        if num_axes != -1:
            end_axis = start_axis + num_axes
        num_axes_replaced = end_axis - start_axis
        num_axes_retained = len(in_shape) - num_axes_replaced
        for i in range(0, start_axis):
            out_shape.append(in_shape[i])
        for dim in shape:
            out_shape.append(dim)
        for i in range(end_axis, len(in_shape)):
            out_shape.append(in_shape[i])
        for copy_axis in copy_axes:
            out_shape[start_axis + copy_axis] = in_shape[start_axis + copy_axis]
        if inferred_axis >= 0:
            explicit_count = constant_count
            for i in range(0, start_axis):
                explicit_count *= in_shape[i]
            for i in range(end_axis, len(in_shape)):
                explicit_count *= in_shape[i]
            for copy_axis in copy_axes:
                explicit_count *= in_shape[start_axis + copy_axis]
            bottom_count = 1
            for dim in in_shape:
                bottom_count *= dim
            out_shape[start_axis + inferred_axis] = bottom_count / explicit_count
        self.blobs[tops[0]] = {'shape': out_shape}

    def add_scale(self, name, bottoms, tops, axis=1, num_axes=1, bias_term=False):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Scale', bottoms, tops)

        if axis != 1:
            layer_param.bias_param.axis = axis
        if num_axes != 1:
            layer_param.bias_param.num_axes = num_axes
        if bias_term != False:
            layer_param.scale_param.bias_term = bias_term

        in_shape = self.blobs[bottoms[0]]['shape']
        self.blobs[tops[0]] = {'shape': in_shape}

        if self.blob_shape:
            scale_blob = layer_param.blobs.add()
            scale_blob.shape.dim.append(in_shape[axis])

            if bias_term != False:
                bias_blob = layer_param.blobs.add()
                bias_blob.shape.dim.append(in_shape[axis])

    def add_softmax(self, name, bottoms, tops, axis=1):
        layer_param = self.net_param.layer.add()
        self.add_common(layer_param, name, 'Softmax', bottoms, tops)

        if axis != 1:
            layer_param.bias_param.axis = axis

        in_shape = self.blobs[bottoms[0]]['shape']
        self.blobs[tops[0]] = {'shape': in_shape}

    def find_layer_by_name(self, name):
        for layer in self.net_param.layer:
            if layer.name == name:
                return layer

    def find_blob_by_name(self, name):
        return self.blobs[name]

    def WriteProtoToTxt(self, file_path):
        with open(file_path, 'w') as proto_file:
            text_format.PrintMessage(self.net_param, proto_file)

    def WriteProtoToBinary(self, file_path):
        with open(file_path, 'wb') as proto_file:
            proto_file.write(self.net_param.SerializeToString())
