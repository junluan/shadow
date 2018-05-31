from __future__ import print_function
from __future__ import division

import copy
import math
from shadow.net_spec import Shadow


def get_arg(shadow_op, arg_name, arg_type, default_value):
    for arg in shadow_op.arg:
        if arg.name == arg_name:
            if arg_type == 's_i':
                return arg.s_i
            elif arg_type == 's_f':
                return arg.s_f
            elif arg_type == 's_s':
                return arg.s_s
            elif arg_type == 'v_i':
                return arg.v_i
            elif arg_type == 'v_f':
                return arg.v_f
            elif arg_type == 'v_s':
                return arg.v_s
    return default_value


def convert_batch_norm(conv_op, bn_op, scale_op, merged_net, copy_params):
    merged_op = merged_net.add_op()
    merged_op.name = conv_op.name
    merged_op.type = conv_op.type
    merged_op.top.extend(scale_op.top)
    merged_op.bottom.extend(conv_op.bottom)

    for arg in conv_op.arg:
        if arg.name != 'bias_term':
            merged_op.arg.add().CopyFrom(arg)

    if not copy_params:
        return

    has_bias = get_arg(conv_op, 'bias_term', 's_i', 1)
    weight = copy.deepcopy(conv_op.blobs[0].data_f)
    out_c = conv_op.blobs[0].shape[0]
    bias = [0 for i in range(0, out_c)]
    if has_bias:
        bias = copy.deepcopy(conv_op.blobs[1].data_f)

    bn_mean = copy.deepcopy(bn_op.blobs[0].data_f)
    bn_var = copy.deepcopy(bn_op.blobs[1].data_f)
    bn_scale = bn_op.blobs[2].data_f[0]
    if bn_scale != 0:
        bn_scale = 1 / bn_scale
    for i in range(0, len(bn_mean)):
        bn_mean[i] *= bn_scale
        bn_var[i] = 1 / math.sqrt(abs(bn_var[i]) * bn_scale + 0.000001)

    scale_scale = copy.deepcopy(scale_op.blobs[0].data_f)
    scale_bias = copy.deepcopy(scale_op.blobs[1].data_f)
    spatial = int(len(weight) / out_c)
    for i in range(0, len(weight)):
        c = int(i / spatial)
        weight[i] *= bn_var[c] * scale_scale[c]
    for i in range(0, len(bias)):
        bias[i] = (bias[i] - bn_mean[i]) * bn_var[i] * scale_scale[i] + scale_bias[i]

    merge_weight_blob = merged_op.blobs.add()
    merge_bias_blob = merged_op.blobs.add()
    merge_weight_blob.shape.extend(conv_op.blobs[0].shape)
    merge_weight_blob.data_f.extend(weight)
    merge_bias_blob.shape.append(out_c)
    merge_bias_blob.data_f.extend(bias)


def convert_activate(conv_op, ac_op, merged_net):
    merged_op = merged_net.add_op()
    merged_op.name = conv_op.name
    merged_op.type = conv_op.type
    merged_op.top.extend(ac_op.top)
    merged_op.bottom.extend(conv_op.bottom)

    for blob in conv_op.blobs:
        merged_op.blobs.add().CopyFrom(blob)

    for arg in conv_op.arg:
        merged_op.arg.add().CopyFrom(arg)

    bias_arg = merged_op.arg.add()
    bias_arg.name = 'type'
    bias_arg.s_i = 1


def MergeBatchNorm(shadow_net, copy_params):
    merged_net = Shadow()
    for n, ori_net in enumerate(shadow_net.get_nets()):
        merged_net.set_net(n)
        merged_net.set_net_name(ori_net.name)
        merged_net.set_net_num_class(ori_net.num_class)
        merged_net.set_net_out_blob(ori_net.out_blob)
        merged_net.copy_net_arg(ori_net.arg)
        op_size = len(ori_net.op)
        o = 0
        while o < op_size:
            op = ori_net.op[o]
            if 'Conv' in op.type:
                bn_index, scale_index = o + 1, o + 2
                has_bn, has_scale = False, False
                if bn_index < op_size:
                    has_bn = ori_net.op[bn_index].type == 'BatchNorm'
                if scale_index < op_size:
                    has_scale = ori_net.op[scale_index].type == 'Scale'
                if has_bn and has_scale:
                    convert_batch_norm(op, ori_net.op[bn_index], ori_net.op[scale_index], merged_net, copy_params)
                    o += 3
                    continue
            merged_net.add_op().CopyFrom(op)
            o += 1
    return merged_net


def MergeActivate(shadow_net):
    merged_net = Shadow()
    for n, ori_net in enumerate(shadow_net.get_nets()):
        merged_net.set_net(n)
        merged_net.set_net_name(ori_net.name)
        merged_net.set_net_num_class(ori_net.num_class)
        merged_net.set_net_out_blob(ori_net.out_blob)
        merged_net.copy_net_arg(ori_net.arg)
        op_size = len(ori_net.op)
        o = 0
        while o < op_size:
            op = ori_net.op[o]
            if 'Conv' in op.type:
                ac_index = o + 1
                has_ac = False
                if ac_index < op_size:
                    ac_op = ori_net.op[ac_index]
                    has_ac = ac_op.type == 'Activate' and get_arg(ac_op, 'type', 's_i', -1) == 1
                if has_ac:
                    convert_activate(op, ori_net.op[ac_index], merged_net)
                    o += 2
                    continue
            merged_net.add_op().CopyFrom(op)
            o += 1
    return merged_net


def Merge(shadow_net, copy_params, merge_activate=True):
    merged_bn_net = MergeBatchNorm(shadow_net, copy_params)
    if merge_activate:
        return MergeActivate(merged_bn_net)
    else:
        return merged_bn_net
