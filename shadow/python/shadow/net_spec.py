from __future__ import print_function

from google.protobuf import text_format
from proto import shadow_pb2


class Shadow(object):
    def __init__(self, name):
        self.meta_net_param = shadow_pb2.MetaNetParam()
        self.meta_net_param.name = name
        self.net_param = {}
        self.net_index = -1

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

    def get_meta_net_name(self):
        if self.meta_net_param.HasField('name'):
            return self.meta_net_param.name
        else:
            return ''

    def get_meta_net_network(self):
        return self.meta_net_param.network

    def get_meta_net_arg(self):
        return self.meta_net_param.arg

    def copy_meta_net_arg(self, arg):
        for ar in arg:
            self.meta_net_param.arg.add().CopyFrom(ar)

    def set_net(self, index):
        for i in range(len(self.meta_net_param.network), index + 1):
            self.meta_net_param.network.add()
        self.net_param = self.meta_net_param.network[index]
        self.net_index = index

    def get_net(self, index):
        assert index >= 0
        assert len(self.meta_net_param.network) > index
        return self.meta_net_param.network[index]

    def set_net_name(self, name):
        self.get_net(self.net_index).name = name

    def get_net_name(self):
        return self.get_net(self.net_index).name

    def get_net_blob(self):
        return self.get_net(self.net_index).blob

    def get_net_op(self):
        return self.get_net(self.net_index).op

    def set_net_arg(self, args):
        for arg_name in args:
            self.set_arg(self.get_net(self.net_index), arg_name[:-4], args[arg_name], arg_name[-3:])

    def get_net_arg(self):
        return self.get_net(self.net_index).arg

    def copy_net_blob(self, blob):
        for b in blob:
            self.get_net(self.net_index).blob.add().CopyFrom(b)

    def copy_net_op(self, op):
        for o in op:
            self.get_net(self.net_index).op.add().CopyFrom(o)

    def copy_net_arg(self, arg):
        for ar in arg:
            self.get_net(self.net_index).arg.add().CopyFrom(ar)

    def add_blob(self):
        return self.get_net(self.net_index).blob.add()

    def add_op(self):
        return self.get_net(self.net_index).op.add()

    def add_input(self, name, bottoms, tops, shapes):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Input', bottoms, tops)

        if len(tops) == len(shapes):
            for n in range(0, len(tops)):
                self.set_arg(op_param, tops[n], shapes[n], 'v_i')
        elif len(shapes) == 1:
            for n in range(0, len(tops)):
                self.set_arg(op_param, tops[n], shapes[0], 'v_i')
        else:
            print('No input shape, must be supplied manually')

    def add_activate(self, name, bottoms, tops, activate_type='Relu', slope=0.1, channel_shared=False):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Activate', bottoms, tops)

        if activate_type == 'PRelu':
            self.set_arg(op_param, 'type', 0, 's_i')
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

    def add_axpy(self, name, bottoms, tops):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Axpy', bottoms, tops)

    def add_batch_norm(self, name, bottoms, tops, use_global_stats=True, eps=1e-5):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'BatchNorm', bottoms, tops)

        self.set_arg(op_param, 'use_global_stats', use_global_stats, 's_i')
        self.set_arg(op_param, 'eps', eps, 's_f')

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

    def add_concat(self, name, bottoms, tops, axis=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Concat', bottoms, tops)

        self.set_arg(op_param, 'axis', axis, 's_i')

    def add_connected(self, name, bottoms, tops, num_output, bias_term=True, transpose=False):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Connected', bottoms, tops)

        self.set_arg(op_param, 'num_output', num_output, 's_i')
        self.set_arg(op_param, 'bias_term', bias_term, 's_i')
        self.set_arg(op_param, 'transpose', transpose, 's_i')

    def add_conv(self, name, bottoms, tops, num_output, kernel_size, stride=1, pad=0, dilation=1, bias_term=True, group=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Conv', bottoms, tops)

        self.set_arg(op_param, 'num_output', num_output, 's_i')
        self.set_arg(op_param, 'kernel_size', kernel_size, 's_i')
        self.set_arg(op_param, 'stride', stride, 's_i')
        self.set_arg(op_param, 'pad', pad, 's_i')
        self.set_arg(op_param, 'dilation', dilation, 's_i')
        self.set_arg(op_param, 'bias_term', bias_term, 's_i')
        self.set_arg(op_param, 'group', group, 's_i')

    def add_deconv(self, name, bottoms, tops, num_output, kernel_size, stride=1, pad=0, dilation=1, bias_term=True, group=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Deconv', bottoms, tops)

        self.set_arg(op_param, 'num_output', num_output, 's_i')
        self.set_arg(op_param, 'kernel_size', kernel_size, 's_i')
        self.set_arg(op_param, 'stride', stride, 's_i')
        self.set_arg(op_param, 'pad', pad, 's_i')
        self.set_arg(op_param, 'dilation', dilation, 's_i')
        self.set_arg(op_param, 'bias_term', bias_term, 's_i')
        self.set_arg(op_param, 'group', group, 's_i')

    def add_deformable_conv(self, name, bottoms, tops, num_output, kernel_size, stride=1, pad=0, dilation=1, bias_term=True, group=1, deformable_group=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'DeformableConv', bottoms, tops)

        self.set_arg(op_param, 'num_output', num_output, 's_i')
        self.set_arg(op_param, 'kernel_size', kernel_size, 's_i')
        self.set_arg(op_param, 'stride', stride, 's_i')
        self.set_arg(op_param, 'pad', pad, 's_i')
        self.set_arg(op_param, 'dilation', dilation, 's_i')
        self.set_arg(op_param, 'bias_term', bias_term, 's_i')
        self.set_arg(op_param, 'group', group, 's_i')
        self.set_arg(op_param, 'deformable_group', deformable_group, 's_i')

    def add_deformable_psroi_pooling(self, name, bottoms, tops, output_dim, group_size, pooled_size, part_size, sample_per_part, spatial_scale, trans_std, no_trans=False):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'DeformablePSROIPooling', bottoms, tops)

        self.set_arg(op_param, 'output_dim', output_dim, 's_i')
        self.set_arg(op_param, 'group_size', group_size, 's_i')
        self.set_arg(op_param, 'pooled_size', pooled_size, 's_i')
        self.set_arg(op_param, 'part_size', part_size, 's_i')
        self.set_arg(op_param, 'sample_per_part', sample_per_part, 's_i')
        self.set_arg(op_param, 'spatial_scale', spatial_scale, 's_f')
        self.set_arg(op_param, 'trans_std', trans_std, 's_f')
        self.set_arg(op_param, 'no_trans', no_trans, 's_i')

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

    def add_flatten(self, name, bottoms, tops, axis=1, end_axis=-1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Flatten', bottoms, tops)

        self.set_arg(op_param, 'axis', axis, 's_i')
        self.set_arg(op_param, 'end_axis', end_axis, 's_i')

    def add_lrn(self, name, bottoms, tops, local_size=5, alpha=1, beta=0.75, norm_region='AcrossChannels', k=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'LRN', bottoms, tops)

        self.set_arg(op_param, 'local_size', local_size, 's_i')
        self.set_arg(op_param, 'alpha', alpha, 's_f')
        self.set_arg(op_param, 'beta', beta, 's_f')
        if norm_region == 'AcrossChannels':
            self.set_arg(op_param, 'norm_region', 0, 's_i')
        else:
            raise ValueError('Unsupported norm region type', norm_region)
        self.set_arg(op_param, 'k', k, 's_f')

    def add_normalize(self, name, bottoms, tops, across_spatial=True, channel_shared=True):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Normalize', bottoms, tops)

        self.set_arg(op_param, 'across_spatial', across_spatial, 's_i')
        self.set_arg(op_param, 'channel_shared', channel_shared, 's_i')

    def add_permute(self, name, bottoms, tops, order=None):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Permute', bottoms, tops)

        if order is not None:
            self.set_arg(op_param, 'order', order, 'v_i')

    def add_pooling(self, name, bottoms, tops, pool, kernel_size, stride, pad, global_pooling=False, full_pooling=True):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Pooling', bottoms, tops)

        if pool == 'Max':
            self.set_arg(op_param, 'pool', 0, 's_i')
        else:
            self.set_arg(op_param, 'pool', 1, 's_i')
        self.set_arg(op_param, 'kernel_size', kernel_size, 'v_i')
        self.set_arg(op_param, 'stride', stride, 'v_i')
        self.set_arg(op_param, 'pad', pad, 'v_i')
        self.set_arg(op_param, 'global_pooling', global_pooling, 's_i')
        self.set_arg(op_param, 'full_pooling', full_pooling, 's_i')

    def add_prior_box(self, name, bottoms, tops, min_size=None, max_size=None, aspect_ratio=None, flip=True, clip=True, variance=None, step=0, offset=0.5):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'PriorBox', bottoms, tops)

        if min_size is not None:
            self.set_arg(op_param, 'min_size', min_size, 'v_f')
        if max_size is not None:
            self.set_arg(op_param, 'max_size', max_size, 'v_f')
        if aspect_ratio is not None:
            self.set_arg(op_param, 'aspect_ratio', aspect_ratio, 'v_f')
        self.set_arg(op_param, 'flip', flip, 's_i')
        self.set_arg(op_param, 'clip', clip, 's_i')
        if variance is not None:
            self.set_arg(op_param, 'variance', variance, 'v_f')
        self.set_arg(op_param, 'step', step, 's_f')
        self.set_arg(op_param, 'offset', offset, 's_f')

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

    def add_psroi_pooling(self, name, bottoms, tops, output_dim, group_size, spatial_scale):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'PSROIPooling', bottoms, tops)

        self.set_arg(op_param, 'output_dim', output_dim, 's_i')
        self.set_arg(op_param, 'group_size', group_size, 's_i')
        self.set_arg(op_param, 'spatial_scale', spatial_scale, 's_f')

    def add_reorg(self, name, bottoms, tops, stride=2):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Reorg', bottoms, tops)

        self.set_arg(op_param, 'stride', stride, 's_i')

    def add_reshape(self, name, bottoms, tops, shape=None, axis=0, num_axes=-1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Reshape', bottoms, tops)

        if shape is not None:
            self.set_arg(op_param, 'shape', shape, 'v_i')
        self.set_arg(op_param, 'axis', axis, 's_i')
        self.set_arg(op_param, 'num_axes', num_axes, 's_i')

    def add_roi_pooling(self, name, bottoms, tops, pooled_h, pooled_w, spatial_scale):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'ROIPooling', bottoms, tops)

        self.set_arg(op_param, 'pooled_h', pooled_h, 's_i')
        self.set_arg(op_param, 'pooled_w', pooled_w, 's_i')
        self.set_arg(op_param, 'spatial_scale', spatial_scale, 's_f')

    def add_scale(self, name, bottoms, tops, axis=1, has_scale=True, has_bias=True, scale_value=None, bias_value=None):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Scale', bottoms, tops)

        self.set_arg(op_param, 'axis', axis, 's_i')
        self.set_arg(op_param, 'has_scale', has_scale, 's_i')
        self.set_arg(op_param, 'has_bias', has_bias, 's_i')
        if scale_value is not None:
            self.set_arg(op_param, 'scale_value', scale_value, 'v_f')
        if bias_value is not None:
            self.set_arg(op_param, 'bias_value', bias_value, 'v_f')

        has_blob = has_scale or has_bias
        has_scalar = scale_value is not None or bias_value is not None
        assert has_blob != has_scalar

    def add_softmax(self, name, bottoms, tops, axis=1):
        op_param = self.net_param.op.add()
        self.add_common(op_param, name, 'Softmax', bottoms, tops)

        self.set_arg(op_param, 'axis', axis, 's_i')

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

    def find_op_by_name(self, name):
        for op in self.get_net(self.net_index).op:
            if op.name == name:
                return op

    def write_proto_to_txt(self, file_path, net_index=-1):
        with open(file_path, 'w') as proto_file:
            if net_index >= 0:
                text_format.PrintMessage(self.get_net(net_index), proto_file)
            else:
                text_format.PrintMessage(self.meta_net_param, proto_file)

    def write_proto_to_binary(self, file_path, net_index=-1):
        with open(file_path, 'wb') as proto_file:
            if net_index >= 0:
                proto_file.write(self.get_net(net_index).SerializeToString())
            else:
                proto_file.write(self.meta_net_param.SerializeToString())
