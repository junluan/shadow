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


def get_blob(network, blob_name):
    for blob in network.blob:
        if blob_name == blob.name:
            return blob
    return None


def convert_batch_norm(o, ori_net, merged_net, copy_params):
    conv_op = ori_net.op[o]
    bn_op = ori_net.op[o + 1]
    scale_op = ori_net.op[o + 2]

    merged_op = merged_net.add_op()
    merged_op.name = conv_op.name
    merged_op.type = conv_op.type
    merged_op.top.extend(scale_op.top)
    merged_op.bottom.append(conv_op.bottom[0])

    for arg in conv_op.arg:
        if arg.name != 'bias_term':
            merged_op.arg.add().CopyFrom(arg)

    if not copy_params:
        return

    merge_weight_blob = merged_net.add_blob()
    merge_bias_blob = merged_net.add_blob()

    merge_weight_blob.name = merged_op.name + '_merged_weights:0'
    merge_bias_blob.name = merged_op.name + '_merged_weights:1'

    merged_op.bottom.append(merge_weight_blob.name)
    merged_op.bottom.append(merge_bias_blob.name)

    conv_weight_blob = get_blob(ori_net, conv_op.bottom[1])
    assert conv_weight_blob is not None
    merge_weight_blob.shape.extend(conv_weight_blob.shape)

    out_c = get_arg(conv_op, 'num_output', 's_i', 0)
    assert out_c > 0
    merge_bias_blob.shape.append(out_c)

    weight = copy.deepcopy(conv_weight_blob.data_f)
    if get_arg(conv_op, 'bias_term', 's_i', 1):
        conv_bias_blob = get_blob(ori_net, conv_op.bottom[2])
        assert conv_bias_blob is not None
        bias = copy.deepcopy(conv_bias_blob.data_f)
    else:
        bias = [0] * out_c

    eps = get_arg(bn_op, 'eps', 's_f', 1e-5)
    bn_mean_blob = get_blob(ori_net, bn_op.bottom[1])
    bn_var_blob = get_blob(ori_net, bn_op.bottom[2])
    bn_scale_blob = get_blob(ori_net, bn_op.bottom[3])
    assert bn_mean_blob is not None
    assert bn_var_blob is not None
    assert bn_scale_blob is not None
    bn_mean = copy.deepcopy(bn_mean_blob.data_f)
    bn_var = copy.deepcopy(bn_var_blob.data_f)
    bn_scale = bn_scale_blob.data_f[0]
    if bn_scale != 0:
        bn_scale = 1 / bn_scale
    for i in range(0, len(bn_mean)):
        bn_mean[i] *= bn_scale
        bn_var[i] = 1 / math.sqrt(abs(bn_var[i]) * bn_scale + eps)

    scale_scale_blob = get_blob(ori_net, scale_op.bottom[1])
    scale_bias_blob = get_blob(ori_net, scale_op.bottom[2])
    assert scale_scale_blob is not None
    assert scale_bias_blob is not None
    scale_scale = copy.deepcopy(scale_scale_blob.data_f)
    scale_bias = copy.deepcopy(scale_bias_blob.data_f)
    spatial = int(len(weight) / out_c)
    for i in range(0, len(weight)):
        c = int(i / spatial)
        weight[i] *= bn_var[c] * scale_scale[c]
    for i in range(0, len(bias)):
        bias[i] = (bias[i] - bn_mean[i]) * bn_var[i] * scale_scale[i] + scale_bias[i]

    merge_weight_blob.data_f.extend(weight)
    merge_bias_blob.data_f.extend(bias)


def convert_activate(conv_op, ac_op, merged_net):
    merged_op = merged_net.add_op()
    merged_op.name = conv_op.name
    merged_op.type = conv_op.type
    merged_op.top.extend(ac_op.top)
    merged_op.bottom.extend(conv_op.bottom)

    for arg in conv_op.arg:
        merged_op.arg.add().CopyFrom(arg)

    bias_arg = merged_op.arg.add()
    bias_arg.name = 'type'
    bias_arg.s_i = 1


def merge_batchnorm(shadow_net, copy_params):
    merged_net = Shadow(shadow_net.get_meta_net_name())
    for n, ori_net in enumerate(shadow_net.get_meta_net_network()):
        merged_net.set_net(n)
        merged_net.set_net_name(ori_net.name)
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
                    convert_batch_norm(o, ori_net, merged_net, copy_params)
                    o += 3
                    continue
            merged_net.add_op().CopyFrom(op)
            o += 1
            for bottom_name in op.bottom:
                blob = get_blob(ori_net, bottom_name)
                if blob is not None:
                    merged_net.add_blob().CopyFrom(blob)
    return merged_net


def merge_activate(shadow_net):
    merged_net = Shadow(shadow_net.get_meta_net_name())
    for n, ori_net in enumerate(shadow_net.get_meta_net_network()):
        merged_net.set_net(n)
        merged_net.set_net_name(ori_net.name)
        merged_net.copy_net_blob(ori_net.blob)
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


def merge(shadow_net, copy_params):
    merged_bn_net = merge_batchnorm(shadow_net, copy_params)
    merged_net = merge_activate(merged_bn_net)
    return merged_net
