import numpy as np
import networkx as nx
from proto import OpParam


optimizers_list = []


def register_optimizer(find_subgraph_func, conditions):
    def optimizer(optimizer_func):
        optimizers_list.append(lambda graph: optimizer_func(graph, find_subgraph_func(graph, conditions)))
        return optimizer_func
    return optimizer


def add_arg(op_param, arg_name, arg_type, arg_value):
    arg = op_param.arg.add()
    arg.name = arg_name
    if arg_type == 's_f':
        arg.s_f = float(arg_value)
    elif arg_type == 's_i':
        arg.s_i = int(arg_value)
    elif arg_type == 's_s':
        arg.s_s = str(arg_value)
    elif arg_type == 'v_f':
        for v in arg_value:
            arg.v_f.append(float(v))
    elif arg_type == 'v_i':
        for v in arg_value:
            arg.v_i.append(int(v))
    elif arg_type == 'v_s':
        for v in arg_value:
            arg.v_s.append(str(v))
    else:
        raise ValueError('Unknown argument type', arg_type)


def get_arg(op_param, arg_name, arg_type, default_value):
    for arg in op_param.arg:
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


def add_blob(net_param, blob_name, blob_shape, blob_data):
    blob = net_param.blob.add()
    blob.name = blob_name
    blob.shape.extend(blob_shape)
    blob.data_f.extend(blob_data.flatten())


def get_blob(net_param, blob_name):
    for blob in net_param.blob:
        if blob_name == blob.name:
            return blob
    return None


def net_param_to_graph(net_param):
    graph = nx.DiGraph(net_param=net_param)

    for op_param in net_param.op:
        graph.add_node(op_param.name, op_param=op_param)

    inner_node = {}
    for op in net_param.op:
        for bottom in op.bottom:
            if bottom in inner_node:
                graph.add_edge(inner_node[bottom], op.name)
        for top in op.top:
            inner_node[top] = op.name

    return graph


def find_successors(graph, name, succ_names):
    succ_nodes = graph.succ[name]
    if len(succ_nodes) != 1:
        return None
    for succ_name in succ_nodes:
        if succ_name in succ_names:
            return succ_name
    return None


def find_subgraph(graph, conditions):
    sets = {name: set() for name in conditions}
    for node_name in graph.nodes():
        op_param = graph.nodes[node_name]['op_param']
        if op_param.type in sets:
            sets[op_param.type].add(node_name)

    subgraphs = []
    for succ_name in sets[conditions[0]]:
        subgraph = [succ_name]
        for n in range(1, len(conditions)):
            succ_name = find_successors(graph, succ_name, sets[conditions[n]])
            if succ_name is None:
                break
            subgraph.append(succ_name)
        if len(subgraph) == len(conditions):
            subgraphs.append(subgraph)

    return subgraphs


def find_consecutive_subgraph(graph, condition):
    node_list = []
    for node_name in nx.topological_sort(graph):
        op_param = graph.nodes[node_name]['op_param']
        if op_param.type == condition:
            node_list.append(node_name)

    subgraphs = []
    while len(node_list) > 0:
        succ_name = node_list[0]
        subgraph = [succ_name]
        while True:
            succ_name = find_successors(graph, succ_name, node_list)
            if succ_name is not None:
                subgraph.append(succ_name)
            else:
                break
        for name in subgraph:
            node_list.remove(name)
        if len(subgraph) > 1:
            subgraphs.append(subgraph)

    return subgraphs


def fuse_subgraph(graph, subgraph, fused_op_param):
    pred_names = [name for name in graph.pred[subgraph[0]]]
    succ_names = [name for name in graph.succ[subgraph[-1]]]
    graph.remove_nodes_from(subgraph)
    fused_name = fused_op_param.name
    graph.add_node(fused_name, op_param=fused_op_param)
    for name in pred_names:
        graph.add_edge(name, fused_name)
    for name in succ_names:
        graph.add_edge(fused_name, name)


@register_optimizer(find_subgraph, ['Conv', 'Activate'])
def fuse_conv_activate(graph, subgraphs):
    has_operation = False

    for subgraph in subgraphs:
        conv_param, activate_param = [graph.nodes[name]['op_param'] for name in subgraph]

        if get_arg(activate_param, 'type', 's_i', -1) != 1:
            continue

        fused_param = OpParam(name=conv_param.name, type=conv_param.type)
        fused_param.bottom.extend(conv_param.bottom)
        fused_param.top.extend(activate_param.top)

        fused_param.arg.extend(conv_param.arg)
        add_arg(fused_param, 'type', 's_i', 1)

        fuse_subgraph(graph, subgraph, fused_param)

        has_operation = True

    return has_operation


@register_optimizer(find_subgraph, ['Conv', 'BatchNorm', 'Scale'])
@register_optimizer(find_subgraph, ['Deconv', 'BatchNorm', 'Scale'])
def fuse_conv_bn_scale(graph, subgraphs):
    has_operation = False

    net_param = graph.graph['net_param']
    for subgraph in subgraphs:
        conv_param, bn_param, scale_param = [graph.nodes[name]['op_param'] for name in subgraph]

        if get_arg(conv_param, 'type', 's_i', -1) != -1:
            continue

        out_c = get_arg(conv_param, 'num_output', 's_i', 0)
        assert out_c > 0
        has_bias = get_arg(conv_param, 'bias_term', 's_i', 1)
        eps = get_arg(bn_param, 'eps', 's_f', 1e-5)

        conv_weight_blob = get_blob(net_param, conv_param.bottom[1])
        assert conv_weight_blob is not None
        assert len(conv_weight_blob.shape) == 4
        conv_weight = np.asarray(conv_weight_blob.data_f).reshape(conv_weight_blob.shape)
        if has_bias:
            conv_bias_blob = get_blob(net_param, conv_param.bottom[2])
            assert conv_bias_blob is not None
            assert len(conv_bias_blob.data_f) == out_c
            conv_bias = np.asarray(conv_bias_blob.data_f)
        else:
            conv_bias = np.zeros(out_c)

        bn_mean_blob = get_blob(net_param, bn_param.bottom[1])
        bn_var_blob = get_blob(net_param, bn_param.bottom[2])
        assert bn_mean_blob is not None and bn_var_blob is not None
        assert len(bn_mean_blob.data_f) == out_c and len(bn_var_blob.data_f) == out_c
        bn_mean, bn_var = np.asarray(bn_mean_blob.data_f), np.asarray(bn_var_blob.data_f)
        if len(bn_param.bottom) == 4:
            bn_scale_blob = get_blob(net_param, bn_param.bottom[3])
            assert bn_scale_blob is not None
            bn_scale = bn_scale_blob.data_f[0]
            bn_scale = 1. / bn_scale if bn_scale != 0 else bn_scale
            bn_mean *= bn_scale
            bn_var *= bn_scale
        bn_var = 1. / np.sqrt(np.absolute(bn_var) + eps)

        scale_scale_blob = get_blob(net_param, scale_param.bottom[1])
        scale_bias_blob = get_blob(net_param, scale_param.bottom[2])
        assert scale_scale_blob is not None and scale_bias_blob is not None
        scale_scale, scale_bias = np.asarray(scale_scale_blob.data_f), np.asarray(scale_bias_blob.data_f)

        if conv_param.type == 'Conv':
            conv_weight *= bn_var.reshape((out_c, 1, 1, 1)) * scale_scale.reshape((out_c, 1, 1, 1))
        elif conv_param.type == 'Deconv':
            conv_weight *= bn_var.reshape((1, out_c, 1, 1)) * scale_scale.reshape((1, out_c, 1, 1))
        else:
            raise ValueError('Unknown op type', conv_param.type)
        conv_bias = (conv_bias - bn_mean) * bn_var * scale_scale + scale_bias

        merged_weight_name = conv_param.name + '/merged_bn_weights:0'
        merged_bias_name = conv_param.name + '/merged_bn_weights:1'

        add_blob(net_param, merged_weight_name, conv_weight.shape, conv_weight)
        add_blob(net_param, merged_bias_name, conv_bias.shape, conv_bias)

        fused_param = OpParam(name=conv_param.name, type=conv_param.type)
        fused_param.bottom.extend([conv_param.bottom[0], merged_weight_name, merged_bias_name])
        fused_param.top.extend(scale_param.top)

        fused_param.arg.extend([arg for arg in conv_param.arg if arg.name != 'bias_term'])

        fuse_subgraph(graph, subgraph, fused_param)

        has_operation = True

    return has_operation


@register_optimizer(find_subgraph, ['Permute', 'MatMul'])
def fuse_permute_matmul(graph, subgraphs):
    has_operation = False

    for subgraph in subgraphs:
        permute_param, matmul_param = [graph.nodes[name]['op_param'] for name in subgraph]

        order = get_arg(permute_param, 'order', 'v_i', None)
        assert order is not None
        transpose_a = get_arg(matmul_param, 'transpose_a', 's_i', False)
        transpose_b = get_arg(matmul_param, 'transpose_b', 's_i', False)

        num_axes = len(order)
        check_order = list(range(num_axes - 2)) + [num_axes - 1, num_axes - 2]
        if order != check_order:
            continue

        index = list(matmul_param.bottom).index(permute_param.top[0])

        matmul_param.bottom[index] = permute_param.bottom[0]

        matmul_param.ClearField('arg')
        if index == 0:
            add_arg(matmul_param, 'transpose_a', 's_i', not transpose_a)
            add_arg(matmul_param, 'transpose_b', 's_i', transpose_b)
        else:
            add_arg(matmul_param, 'transpose_a', 's_i', transpose_a)
            add_arg(matmul_param, 'transpose_b', 's_i', not transpose_b)

        pred_names = [name for name in graph.pred[subgraph[0]]]
        for name in pred_names:
            graph.add_edge(name, subgraph[1])
        graph.remove_node(subgraph[0])

        has_operation = True

    return has_operation


@register_optimizer(find_consecutive_subgraph, 'Permute')
def fuse_consecutive_permute(graph, subgraphs):
    has_operation = False

    for subgraph in subgraphs:
        permute_params = [graph.nodes[name]['op_param'] for name in subgraph]

        orders = [get_arg(permute_param, 'order', 'v_i', None) for permute_param in permute_params]
        assert None not in orders

        fused_order = range(len(orders[0]))
        for order in orders:
            fused_order = np.choose(order, fused_order)

        fused_param = OpParam(name=permute_params[0].name, type=permute_params[0].type)
        fused_param.bottom.extend(permute_params[0].bottom)
        fused_param.top.extend(permute_params[-1].top)

        add_arg(fused_param, 'order', 'v_i', fused_order)

        fuse_subgraph(graph, subgraph, fused_param)

        has_operation = True

    return has_operation


@register_optimizer(find_consecutive_subgraph, 'Squeeze')
def fuse_consecutive_squeeze(graph, subgraphs):
    has_operation = False

    for subgraph in subgraphs:
        squeeze_params = [graph.nodes[name]['op_param'] for name in subgraph]

        all_axes = [get_arg(squeeze_param, 'axes', 'v_i', None) for squeeze_param in squeeze_params]

        left_axes = range(10)
        for axes in all_axes:
            if axes is None:
                left_axes = []
            else:
                left_axes = [left_axes[d] for d in range(len(left_axes)) if d not in axes]

        if len(left_axes) == 0:
            axes = []
        else:
            axes = [d for d in range(10) if d not in left_axes]

        fused_param = OpParam(name=squeeze_params[0].name, type=squeeze_params[0].type)
        fused_param.bottom.extend(squeeze_params[0].bottom)
        fused_param.top.extend(squeeze_params[-1].top)

        if len(axes) > 0:
            add_arg(fused_param, 'axes', 'v_i', axes)

        fuse_subgraph(graph, subgraph, fused_param)

        has_operation = True

    return has_operation


@register_optimizer(find_subgraph, ['Binary'])
def move_binary_const_to_scalar(graph, subgraphs):
    has_operation = False

    net_param = graph.graph['net_param']
    for subgraph in subgraphs:
        binary_param = graph.nodes[subgraph[0]]['op_param']

        if len(binary_param.bottom) == 1:
            break

        has_weight, weight_index, weight_val = False, -1, None
        for n, bottom in enumerate(binary_param.bottom):
            weight_blob = get_blob(net_param, bottom)
            if weight_blob is not None and len(weight_blob.data_f) == 1:
                has_weight = True
                weight_index = n
                weight_val = weight_blob.data_f[0]
                break
        if not has_weight:
            break

        operation = get_arg(binary_param, 'operation', 's_i', None)
        assert operation is not None

        if operation in [1, 3, 4] and weight_index == 0:
            break

        fused_param = OpParam(name=binary_param.name, type=binary_param.type)
        fused_param.bottom.append(binary_param.bottom[1 - weight_index])
        fused_param.top.extend(binary_param.top)

        add_arg(fused_param, 'operation', 's_i', operation)
        add_arg(fused_param, 'scalar', 's_f', weight_val)

        fuse_subgraph(graph, subgraph, fused_param)

        has_operation = True

    return has_operation


def optimize(network):
    for net_param in network.get_meta_net_network():
        if len(net_param.op) == 0:
            continue

        graph = net_param_to_graph(net_param)

        while True:
            has_operation = False

            for optimizer in optimizers_list:
                has_operation |= optimizer(graph)

            if not has_operation:
                break

        op_params, blobs = [], []
        for node_name in nx.topological_sort(graph):
            op_param = graph.nodes[node_name]['op_param']
            op_params.append(op_param)
            blobs.extend([get_blob(net_param, bottom) for bottom in op_param.bottom if 'weights:' in bottom])

        net_param.ClearField('op')
        net_param.op.extend(op_params)

        net_param.ClearField('blob')
        net_param.blob.extend(blobs)
