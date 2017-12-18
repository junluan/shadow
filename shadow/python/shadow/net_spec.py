from __future__ import print_function

import math
from google.protobuf import text_format
from proto import shadow_pb2


class Shadow(object):
    def __init__(self, blob_shape=True):
        self.meta_net_param = []
        self.net_param = {}
        self.net_index = -1
        self.blobs = []
        self.blob_shape = blob_shape

    @staticmethod
    def set_arg(op_param, arg_name, arg_val, arg_type):
        arg = op_param.arg.add()
        arg.name = arg_name
        if arg_type == 's_f':
            arg.s_f = float(arg_val)
        elif arg_type == 's_i':
            arg.s_i = int(arg_val)
        elif arg_type == 's_s':
            arg.s_s = str(arg_val)
        elif arg_type == 'v_f':
            for v in arg_val:
                arg.v_f.append(float(v))
        elif arg_type == 'v_i':
            for v in arg_val:
                arg.v_i.append(int(v))
        elif arg_type == 'v_s':
            for v in arg_val:
                arg.v_s.append(str(v))
        else:
            raise ValueError('Unknown argument type', arg_type)

    @staticmethod
    def add_common(op_param, op_name, op_type, bottoms, tops):
        op_param.name = op_name
        op_param.type = op_type
        op_param.bottom.extend(bottoms)
        op_param.top.extend(tops)

    def get_nets(self):
        return self.meta_net_param

    def get_net(self, index):
        assert index >= 0
        assert len(self.meta_net_param) > index
        return self.meta_net_param[index]

    def get_net_name(self):
        return self.get_net(self.net_index).name

    def get_net_num_class(self):
        return self.get_net(self.net_index).num_class

    def get_net_out_blob(self):
        return self.get_net(self.net_index).out_blob

    def get_net_op(self):
        return self.get_net(self.net_index).op

    def get_net_arg(self):
        return self.get_net(self.net_index).arg

    def set_net(self, index):
        for i in range(len(self.meta_net_param), index + 1):
            self.meta_net_param.append(shadow_pb2.NetParam())
        self.net_param = self.meta_net_param[index]
        for i in range(len(self.blobs), index + 1):
            self.blobs.append({})
        self.net_index = index

    def set_net_name(self, name):
        self.get_net(self.net_index).name = name

    def set_net_num_class(self, num_class):
        self.get_net(self.net_index).num_class.extend(num_class)

    def set_net_out_blob(self, out_blob):
        self.get_net(self.net_index).out_blob.extend(out_blob)

    def set_net_arg(self, args):
        for arg_name in args:
            self.set_arg(self.get_net(self.net_index), arg_name[:-4], args[arg_name], arg_name[-3:])

    def copy_net_arg(self, arg):
        for ar in arg:
            self.get_net(self.net_index).arg.add().CopyFrom(ar)

    def add_op(self):
        return self.get_net(self.net_index).op.add()

    def add_input(self, name, bottoms, tops, shapes):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Input', bottoms, tops)

        if len(tops) == len(shapes):
            for n in range(0, len(tops)):
                self.set_arg(op_param, tops[n], shapes[n], 'v_i')
                self.blobs[self.net_index][tops[n]] = {'shape': shapes[n]}
        elif len(shapes) == 1:
            for n in range(0, len(tops)):
                self.set_arg(op_param, tops[n], shapes[0], 'v_i')
                self.blobs[self.net_index][tops[n]] = {'shape': shapes[0]}
        else:
            print('No input shape, must be supplied manually')

    def add_activate(self, name, bottoms, tops, activate_type='Relu', slope=0.1, channel_shared=False):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Activate', bottoms, tops)

        if activate_type == 'PRelu':
            self.set_arg(op_param, 'type', 0, 's_i')
            if channel_shared:
                self.set_arg(op_param, 'channel_shared', channel_shared, 's_i')
        elif activate_type == 'Relu':
            self.set_arg(op_param, 'type', 1, 's_i')
        elif activate_type == 'Leaky':
            self.set_arg(op_param, 'type', 2, 's_i')
            self.set_arg(op_param, 'slope', slope, 's_f')
        elif activate_type == 'Sigmoid':
            self.set_arg(op_param, 'type', 3, 's_i')
        elif activate_type == 'SoftPlus':
            self.set_arg(op_param, 'type', 4, 's_i')
        elif activate_type == 'Tanh':
            self.set_arg(op_param, 'type', 5, 's_i')
        else:
            raise ValueError('Unsupported activate type', activate_type)

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}
        if activate_type == 'PRelu' and self.blob_shape:
            slope_blob = op_param.blobs.add()
            if channel_shared:
                slope_blob.shape.extend([1, 1])
            else:
                slope_blob.shape.extend([1, in_shape[1]])

    def add_batch_norm(self, name, bottoms, tops, use_global_stats=True, eps=1e-5):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'BatchNorm', bottoms, tops)

        if not use_global_stats:
            self.set_arg(op_param, 'use_global_stats', use_global_stats, 's_i')
        self.set_arg(op_param, 'eps', eps, 's_f')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}
        if use_global_stats and self.blob_shape:
            mean_blob = op_param.blobs.add()
            mean_blob.shape.extend([in_shape[1]])
            variance_blob = op_param.blobs.add()
            variance_blob.shape.extend([in_shape[1]])
            factor_blob = op_param.blobs.add()
            factor_blob.shape.extend([1])

    def add_bias(self, name, bottoms, tops, axis=1, num_axes=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Bias', bottoms, tops)

        if axis != 1:
            self.set_arg(op_param, 'axis', axis, 's_i')
        if num_axes != 1:
            self.set_arg(op_param, 'num_axes', num_axes, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}
        if self.blob_shape:
            bias_blob = op_param.blobs.add()
            bias_blob.shape.extend([in_shape[axis]])

    def add_binary(self, name, bottoms, tops, operation, scalar=None):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Binary', bottoms, tops)

        if operation == 'Add':
            self.set_arg(op_param, 'operation', 0, 's_i')
        elif operation == 'Sub':
            self.set_arg(op_param, 'operation', 1, 's_i')
        elif operation == 'Mul':
            self.set_arg(op_param, 'operation', 2, 's_i')
        elif operation == 'Div':
            self.set_arg(op_param, 'operation', 3, 's_i')
        elif operation == 'Pow':
            self.set_arg(op_param, 'operation', 4, 's_i')
        elif operation == 'Max':
            self.set_arg(op_param, 'operation', 5, 's_i')
        elif operation == 'Min':
            self.set_arg(op_param, 'operation', 6, 's_i')
        else:
            raise ValueError('Unsupported operation type', operation)
        if scalar is not None:
            self.set_arg(op_param, 'scalar', scalar, 's_f')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}

    def add_concat(self, name, bottoms, tops, axis=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Concat', bottoms, tops)

        if axis != 1:
            self.set_arg(op_param, 'axis', axis, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[axis] = 0
        for bottom in bottoms:
            in_shape = self.blobs[self.net_index][bottom]['shape']
            out_shape[axis] += in_shape[axis]
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_connected(self, name, bottoms, tops, num_output, bias_term=True, transpose=False):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Connected', bottoms, tops)

        self.set_arg(op_param, 'num_output', num_output, 's_i')
        if not bias_term:
            self.set_arg(op_param, 'bias_term', bias_term, 's_i')
        if transpose:
            self.set_arg(op_param, 'transpose', transpose, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        out_shape = [in_shape[0], num_output]
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}
        bottom_num = 1
        for i in range(1, len(in_shape)):
            bottom_num *= in_shape[i]
        if self.blob_shape:
            weight_blob = op_param.blobs.add()
            weight_blob.shape.extend([num_output, bottom_num])
            if bias_term:
                bias_blob = op_param.blobs.add()
                bias_blob.shape.extend([num_output])

    def add_conv(self, name, bottoms, tops, num_output, kernel_size, stride=1, pad=0, dilation=1, bias_term=True, group=1):
        op_param = self.net_param.op.add()
        if num_output != group:
            self.add_common(op_param, name, 'Conv', bottoms, tops)
        else:
            self.add_common(op_param, name, 'DepthwiseConv', bottoms, tops)

        self.set_arg(op_param, 'num_output', num_output, 's_i')
        self.set_arg(op_param, 'kernel_size', kernel_size, 's_i')
        self.set_arg(op_param, 'stride', stride, 's_i')
        self.set_arg(op_param, 'pad', pad, 's_i')
        if dilation != 1:
            self.set_arg(op_param, 'dilation', dilation, 's_i')
        if not bias_term:
            self.set_arg(op_param, 'bias_term', bias_term, 's_i')
        if group != 1:
            self.set_arg(op_param, 'group', group, 's_i')

        def conv_out_size(dim, ks, sd, pa, dila):
            kernel_extent = dila * (ks - 1) + 1
            return (dim + 2 * pa - kernel_extent) / sd + 1

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[1] = num_output
        out_shape[2] = conv_out_size(
            in_shape[2], kernel_size, stride, pad, dilation)
        out_shape[3] = conv_out_size(
            in_shape[3], kernel_size, stride, pad, dilation)
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}
        if self.blob_shape:
            weight_blob = op_param.blobs.add()
            weight_blob.shape.extend([num_output, in_shape[1] / group, kernel_size, kernel_size])
            if bias_term:
                bias_blob = op_param.blobs.add()
                bias_blob.shape.extend([num_output])

    def add_data(self, name, bottoms, tops, mean_value=None, scale_value=None):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Data', bottoms, tops)

        if mean_value is not None:
            self.set_arg(op_param, 'mean_value', mean_value, 'v_f')
        if scale_value is not None:
            self.set_arg(op_param, 'scale_value', scale_value, 'v_f')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}

    def add_deconv(self, name, bottoms, tops, num_output, kernel_size, stride=1, pad=0, dilation=1, bias_term=True, group=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Deconv', bottoms, tops)

        self.set_arg(op_param, 'num_output', num_output, 's_i')
        self.set_arg(op_param, 'kernel_size', kernel_size, 's_i')
        self.set_arg(op_param, 'stride', stride, 's_i')
        self.set_arg(op_param, 'pad', pad, 's_i')
        if dilation != 1:
            self.set_arg(op_param, 'dilation', dilation, 's_i')
        if not bias_term:
            self.set_arg(op_param, 'bias_term', bias_term, 's_i')
        if group != 1:
            self.set_arg(op_param, 'group', group, 's_i')

        def deconv_out_size(dim, ks, sd, pa, dila):
            kernel_extent = dila * (ks - 1) + 1
            return sd * (dim - 1) + kernel_extent - 2 * pa

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[1] = num_output
        out_shape[2] = deconv_out_size(
            in_shape[2], kernel_size, stride, pad, dilation)
        out_shape[3] = deconv_out_size(
            in_shape[3], kernel_size, stride, pad, dilation)
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}
        if self.blob_shape:
            weight_blob = op_param.blobs.add()
            weight_blob.shape.extend([in_shape[1], num_output / group, kernel_size, kernel_size])
            if bias_term:
                bias_blob = op_param.blobs.add()
                bias_blob.shape.extend([num_output])

    def add_deformable_conv(self, name, bottoms, tops, num_output, kernel_size, stride=1, pad=0, dilation=1, bias_term=True, group=1, deformable_group=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'DeformableConv', bottoms, tops)

        self.set_arg(op_param, 'num_output', num_output, 's_i')
        self.set_arg(op_param, 'kernel_size', kernel_size, 's_i')
        self.set_arg(op_param, 'stride', stride, 's_i')
        self.set_arg(op_param, 'pad', pad, 's_i')
        if dilation != 1:
            self.set_arg(op_param, 'dilation', dilation, 's_i')
        if not bias_term:
            self.set_arg(op_param, 'bias_term', bias_term, 's_i')
        if group != 1:
            self.set_arg(op_param, 'group', group, 's_i')
        if deformable_group != 1:
            self.set_arg(op_param, 'deformable_group', deformable_group, 's_i')

        def deformable_conv_out_size(dim, ks, sd, pa, dila):
            kernel_extent = dila * (ks - 1) + 1
            return (dim + 2 * pa - kernel_extent) / sd + 1

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[1] = num_output
        out_shape[2] = deformable_conv_out_size(
            in_shape[2], kernel_size, stride, pad, dilation)
        out_shape[3] = deformable_conv_out_size(
            in_shape[3], kernel_size, stride, pad, dilation)
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}
        if self.blob_shape:
            weight_blob = op_param.blobs.add()
            weight_blob.shape.extend([num_output, in_shape[1] / group, kernel_size, kernel_size])
            if bias_term:
                bias_blob = op_param.blobs.add()
                bias_blob.shape.extend([num_output])

    def add_deformable_psroi_pooling(self, name, bottoms, tops, output_dim, group_size, pooled_size, part_size, sample_per_part, spatial_scale, trans_std, no_trans=False):
        op_param = self.net_param.op.add()
        self.add_common(
            op_param, name, 'DeformablePSROIPooling', bottoms, tops)

        self.set_arg(op_param, 'output_dim', output_dim, 's_i')
        self.set_arg(op_param, 'group_size', group_size, 's_i')
        self.set_arg(op_param, 'pooled_size', pooled_size, 's_i')
        self.set_arg(op_param, 'part_size', part_size, 's_i')
        self.set_arg(op_param, 'sample_per_part', sample_per_part, 's_i')
        self.set_arg(op_param, 'spatial_scale', spatial_scale, 's_f')
        self.set_arg(op_param, 'trans_std', trans_std, 's_f')
        self.set_arg(op_param, 'no_trans', no_trans, 's_i')

        roi_shape = self.blobs[self.net_index][bottoms[1]]['shape']
        out_shape = [roi_shape[0], output_dim, pooled_size, pooled_size]
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_eltwise(self, name, bottoms, tops, operation, coeff=None):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Eltwise', bottoms, tops)

        if operation == 'Prod':
            self.set_arg(op_param, 'operation', 0, 's_i')
        elif operation == 'Sum':
            self.set_arg(op_param, 'operation', 1, 's_i')
        elif operation == 'Max':
            self.set_arg(op_param, 'operation', 2, 's_i')
        else:
            raise ValueError('Unsupported operation type', operation)
        if coeff is not None:
            self.set_arg(op_param, 'coeff', coeff, 'v_f')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}

    def add_flatten(self, name, bottoms, tops, axis=1, end_axis=-1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Flatten', bottoms, tops)

        if axis != 1:
            self.set_arg(op_param, 'axis', axis, 's_i')
        if end_axis != -1:
            self.set_arg(op_param, 'end_axis', end_axis, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
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
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_lrn(self, name, bottoms, tops, local_size=5, alpha=1, beta=0.75, norm_region='AcrossChannels', k=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'LRN', bottoms, tops)

        if local_size != 5:
            self.set_arg(op_param, 'local_size', local_size, 's_i')
        if alpha != 1:
            self.set_arg(op_param, 'alpha', alpha, 's_f')
        if beta != 0.75:
            self.set_arg(op_param, 'beta', beta, 's_f')
        if norm_region == 'AcrossChannels':
            self.set_arg(op_param, 'norm_region', 0, 's_i')
        else:
            raise ValueError('Unsupported norm region type', norm_region)
        if k != 1:
            self.set_arg(op_param, 'k', k, 's_f')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}

    def add_normalize(self, name, bottoms, tops, across_spatial=True, channel_shared=True):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Normalize', bottoms, tops)

        if not across_spatial:
            self.set_arg(op_param, 'across_spatial', across_spatial, 's_i')
        if not channel_shared:
            self.set_arg(op_param, 'channel_shared', channel_shared, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}
        if self.blob_shape:
            scale_blob = op_param.blobs.add()
            if channel_shared:
                scale_blob.shape.extend([1])
            else:
                scale_blob.shape.extend([in_shape[1]])

    def add_permute(self, name, bottoms, tops, order=None):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Permute', bottoms, tops)

        if order is not None:
            self.set_arg(op_param, 'order', order, 'v_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        out_shape = []
        for o in order:
            out_shape.append(in_shape[o])
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_pooling(self, name, bottoms, tops, pool, kernel_size, stride=1, pad=0, global_pooling=False, full_pooling=True):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Pooling', bottoms, tops)

        if pool == 'Max':
            self.set_arg(op_param, 'pool', 0, 's_i')
        else:
            self.set_arg(op_param, 'pool', 1, 's_i')
        self.set_arg(op_param, 'kernel_size', kernel_size, 's_i')
        self.set_arg(op_param, 'stride', stride, 's_i')
        self.set_arg(op_param, 'pad', pad, 's_i')
        if global_pooling:
            self.set_arg(op_param, 'global_pooling', global_pooling, 's_i')
        if not full_pooling:
            self.set_arg(op_param, 'full_pooling', full_pooling, 's_i')

        def pooling_out_size(dim, ks, sd, pa):
            if full_pooling:
                return int(math.ceil(float(dim + 2 * pa - ks) / sd)) + 1
            else:
                return int(math.floor(float(dim + 2 * pa - ks) / sd)) + 1

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        out_shape = in_shape[:]
        if global_pooling:
            kernel_size = in_shape[2]
            stride = 1
            pad = 0
        out_shape[2] = pooling_out_size(in_shape[2], kernel_size, stride, pad)
        out_shape[3] = pooling_out_size(in_shape[3], kernel_size, stride, pad)
        if pad:
            if (out_shape[2] - 1) * stride >= in_shape[2] + pad:
                out_shape[2] = out_shape[2] - 1
            if (out_shape[3] - 1) * stride >= in_shape[3] + pad:
                out_shape[3] = out_shape[3] - 1
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_prior_box(self, name, bottoms, tops, min_size=None, max_size=None, aspect_ratio=None, flip=True, clip=True, variance=None, step=0, offset=0.5):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'PriorBox', bottoms, tops)

        if min_size is not None:
            self.set_arg(op_param, 'min_size', min_size, 'v_f')
        if max_size is not None:
            self.set_arg(op_param, 'max_size', max_size, 'v_f')
        if aspect_ratio is not None:
            self.set_arg(op_param, 'aspect_ratio', aspect_ratio, 'v_f')
        if not flip:
            self.set_arg(op_param, 'flip', flip, 's_i')
        if not clip:
            self.set_arg(op_param, 'clip', clip, 's_i')
        if variance is not None:
            self.set_arg(op_param, 'variance', variance, 'v_f')
        if step != 0:
            self.set_arg(op_param, 'step', step, 's_f')
        if offset != 0.5:
            self.set_arg(op_param, 'offset', offset, 's_f')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        num_asp = len(aspect_ratio)
        if 1 in aspect_ratio:
            num_asp -= 1
        if flip:
            num_asp *= 2
        num_priors = 1 + num_asp
        out_shape = [1, 2, in_shape[2] * in_shape[3] * num_priors * 4]
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_proposal(self, name, bottoms, tops, feat_stride=16, pre_nms_top_n=6000, post_nms_top_n=300, min_size=16, nms_thresh=0.7, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Proposal', bottoms, tops)

        self.set_arg(op_param, 'feat_stride', feat_stride, 's_i')
        self.set_arg(op_param, 'pre_nms_top_n', pre_nms_top_n, 's_i')
        self.set_arg(op_param, 'post_nms_top_n', post_nms_top_n, 's_i')
        self.set_arg(op_param, 'min_size', min_size, 's_i')
        self.set_arg(op_param, 'nms_thresh', nms_thresh, 's_f')
        self.set_arg(op_param, 'ratios', ratios, 'v_f')
        self.set_arg(op_param, 'scales', scales, 'v_f')

        self.blobs[self.net_index][tops[0]] = {'shape': [post_nms_top_n, 5]}

    def add_psroi_pooling(self, name, bottoms, tops, output_dim, group_size, spatial_scale):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'PSROIPooling', bottoms, tops)

        self.set_arg(op_param, 'output_dim', output_dim, 's_i')
        self.set_arg(op_param, 'group_size', group_size, 's_i')
        self.set_arg(op_param, 'spatial_scale', spatial_scale, 's_f')

        roi_shape = self.blobs[self.net_index][bottoms[1]]['shape']
        out_shape = [roi_shape[0], output_dim, group_size, group_size]
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_reorg(self, name, bottoms, tops, stride=2):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Reorg', bottoms, tops)

        self.set_arg(op_param, 'stride', stride, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        out_shape = in_shape[:]
        out_shape[1] = in_shape[1] * stride * stride
        out_shape[2] = in_shape[2] / stride
        out_shape[3] = in_shape[3] / stride
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_reshape(self, name, bottoms, tops, shape=None, axis=0, num_axes=-1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Reshape', bottoms, tops)

        if shape is not None:
            self.set_arg(op_param, 'shape', shape, 'v_i')
        if axis != 0:
            self.set_arg(op_param, 'axis', axis, 's_i')
        if num_axes != -1:
            self.set_arg(op_param, 'num_axes', num_axes, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
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
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_roi_pooling(self, name, bottoms, tops, pooled_h, pooled_w, spatial_scale):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'ROIPooling', bottoms, tops)

        self.set_arg(op_param, 'pooled_h', pooled_h, 's_i')
        self.set_arg(op_param, 'pooled_w', pooled_w, 's_i')
        self.set_arg(op_param, 'spatial_scale', spatial_scale, 's_f')

        fea_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        roi_shape = self.blobs[self.net_index][bottoms[1]]['shape']
        out_shape = [roi_shape[0], fea_shape[1], pooled_h, pooled_w]
        self.blobs[self.net_index][tops[0]] = {'shape': out_shape}

    def add_scale(self, name, bottoms, tops, axis=1, num_axes=1, bias_term=False):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Scale', bottoms, tops)

        if axis != 1:
            self.set_arg(op_param, 'axis', axis, 's_i')
        if num_axes != 1:
            self.set_arg(op_param, 'num_axes', num_axes, 's_i')
        if bias_term:
            self.set_arg(op_param, 'bias_term', bias_term, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}
        if self.blob_shape:
            scale_blob = op_param.blobs.add()
            scale_blob.shape.extend([in_shape[axis]])
            if bias_term:
                bias_blob = op_param.blobs.add()
                bias_blob.shape.extend([in_shape[axis]])

    def add_softmax(self, name, bottoms, tops, axis=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Softmax', bottoms, tops)

        if axis != 1:
            self.set_arg(op_param, 'axis', axis, 's_i')

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}

    def add_unary(self, name, bottoms, tops, operation):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Binary', bottoms, tops)

        if operation == 'Abs':
            self.set_arg(op_param, 'operation', 0, 's_i')
        elif operation == 'Square':
            self.set_arg(op_param, 'operation', 1, 's_i')
        elif operation == 'Sqrt':
            self.set_arg(op_param, 'operation', 2, 's_i')
        elif operation == 'Log':
            self.set_arg(op_param, 'operation', 3, 's_i')
        elif operation == 'Exp':
            self.set_arg(op_param, 'operation', 4, 's_i')
        elif operation == 'Sin':
            self.set_arg(op_param, 'operation', 5, 's_i')
        elif operation == 'Cos':
            self.set_arg(op_param, 'operation', 6, 's_i')
        elif operation == 'Tan':
            self.set_arg(op_param, 'operation', 7, 's_i')
        elif operation == 'Asin':
            self.set_arg(op_param, 'operation', 8, 's_i')
        elif operation == 'Acos':
            self.set_arg(op_param, 'operation', 9, 's_i')
        elif operation == 'Atan':
            self.set_arg(op_param, 'operation', 10, 's_i')
        elif operation == 'Floor':
            self.set_arg(op_param, 'operation', 11, 's_i')
        elif operation == 'Ceil':
            self.set_arg(op_param, 'operation', 12, 's_i')
        else:
            raise ValueError('Unsupported operation type', operation)

        in_shape = self.blobs[self.net_index][bottoms[0]]['shape']
        self.blobs[self.net_index][tops[0]] = {'shape': in_shape}

    def find_op_by_name(self, name):
        for op in self.get_net(self.net_index).op:
            if op.name == name:
                return op

    def write_proto_to_txt(self, file_path, net_index=-1):
        with open(file_path, 'w') as proto_file:
            text_format.PrintMessage(self.get_net(net_index), proto_file)

    def write_proto_to_binary(self, file_path, net_index=-1):
        with open(file_path, 'wb') as proto_file:
            proto_file.write(self.get_net(net_index).SerializeToString())
