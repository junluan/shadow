from google.protobuf import text_format
from google.protobuf import json_format
from proto import shadow_pb2


def find_replace(string, old_str, new_str):
    for s in old_str:
        string = string.replace(s, new_str)
    return string


def dump_weights(f, weights, weights_format, num_of_line=10):
    for n, w in enumerate(weights):
        if n > 0:
            f.write(',')
        if n > 0 and n % num_of_line == 0:
            f.write('\n')
        f.write(weights_format.format(w))


def get_blobs(net_param, op_param):
    weight_blobs = []
    for blob_name in op_param.bottom:
        for blob in net_param.blob:
            if blob_name == blob.name:
                weight_blobs.append(blob)
    return weight_blobs


def convert_arg(arg):
    arg_ss = '{{{}, '.format(arg.name)
    if arg.HasField('s_i'):
        arg_ss += 's_i, [{}]'.format(arg.s_i)
    elif arg.HasField('s_f'):
        arg_ss += 's_f, [{}]'.format(arg.s_f)
    elif arg.HasField('s_s'):
        arg_ss += 's_s, [{}]'.format(arg.s_s)
    elif len(arg.v_i) > 0:
        arg_ss += 'v_i, [{}]'.format(' # '.join([str(v) for v in arg.v_i]))
    elif len(arg.v_f) > 0:
        arg_ss += 'v_f, [{}]'.format(' # '.join([str(v) for v in arg.v_f]))
    elif len(arg.v_s) > 0:
        arg_ss += 'v_s, [{}]'.format(' # '.join([str(v) for v in arg.v_s]))
    arg_ss += '}'
    return arg_ss


def convert_custom(net_param):
    ss = '[{}]\n'.format(net_param.name)
    ss += '[{}]\n'.format('; '.join([convert_arg(arg) for arg in net_param.arg]))

    blob_vec = []
    for blob in net_param.blob:
        blob_type = blob.type if blob.HasField('type') else 'float'
        blob_vec.append('{{{}, {}, [{}]}}'.format(blob.name, blob_type, ' # '.join([str(d) for d in blob.shape])))
    ss += '[{}]\n'.format('; '.join(blob_vec))

    for op_param in net_param.op:
        ss += '[{}] | [{}] | '.format(op_param.type, op_param.name)
        ss += '[{} {} {}] | '.format(len(op_param.bottom), len(op_param.top), len(op_param.arg))
        ss += '[{}] | '.format('; '.join(op_param.bottom))
        ss += '[{}] | '.format('; '.join(op_param.top))
        ss += '[{}]\n'.format('; '.join([convert_arg(arg) for arg in op_param.arg]))

    return ss


def write_defines(net_param, root, model_name):
    class_name = model_name.upper()
    weight_prefix = model_name.lower()
    model_name_cpp = model_name + '.cpp'
    model_name_hpp = model_name + '.hpp'
    model_name_weights_hpp = model_name + '_weights.hpp'

    net = shadow_pb2.NetParam()
    net.CopyFrom(net_param)

    blob_counts, blob_names, blob_types = [], [], []
    for blob in net.blob:
        blob_type = blob.type if blob.HasField('type') else 'float'
        if blob_type == 'float':
            blob_counts.append(str(len(blob.data_f)))
        elif blob_type == 'int':
            blob_counts.append(str(len(blob.data_i)))
        elif blob_type == 'unsigned char':
            assert len(blob.data_b) == 1
            blob_counts.append(str(len(blob.data_b[0])))
        else:
            raise ValueError('Unknown blob type', blob_type)
        blob_names.append('{}_{}_'.format(weight_prefix, find_replace(blob.name, ['/', '-', ':'], '_')))
        blob_types.append(blob_type)
        blob.ClearField('data_f')
        blob.ClearField('data_i')
        blob.ClearField('data_b')

    proto_str = text_format.MessageToString(net)
    json_str = json_format.MessageToJson(net, preserving_proto_field_name=True)
    custom_str = convert_custom(net_param)

    split_count = 10000

    proto_split_off = len(proto_str) % split_count
    proto_split_num = len(proto_str) // split_count + (proto_split_off > 0)
    json_split_off = len(json_str) % split_count
    json_split_num = len(json_str) // split_count + (json_split_off > 0)
    custom_split_off = len(custom_str) % split_count
    custom_split_num = len(custom_str) // split_count + (custom_split_off > 0)

    proto_split_names = ['proto_model_{}_'.format(n) for n in range(proto_split_num)]
    json_split_names = ['json_model_{}_'.format(n) for n in range(json_split_num)]
    custom_split_names = ['custom_model_{}_'.format(n) for n in range(custom_split_num)]

    ########## write network proto definition to cpp ##########
    with open('{}/{}'.format(root, model_name_cpp), 'w') as cpp_file:
        cpp_file.write('#include "{}"\n'.format(model_name_hpp))
        cpp_file.write('#include "{}"\n\n'.format(model_name_weights_hpp))

        cpp_file.write('namespace Shadow {\n\n')

        offset = 0
        for proto_split_name in proto_split_names:
            cpp_file.write('const std::string {} = \nR"({})";\n\n'.format(proto_split_name, proto_str[offset: offset + split_count]))
            offset += split_count
        cpp_file.write('const std::string {}::proto_model_{{\n    {}\n}};\n\n'.format(class_name, ' + '.join(proto_split_names)))

        offset = 0
        for json_split_name in json_split_names:
            cpp_file.write('const std::string {} = \nR"({})";\n\n'.format(json_split_name, json_str[offset: offset + split_count]))
            offset += split_count
        cpp_file.write('const std::string {}::json_model_{{\n    {}\n}};\n\n'.format(class_name, ' + '.join(json_split_names)))

        offset = 0
        for custom_split_name in custom_split_names:
            cpp_file.write('const std::string {} = \nR"({})";\n\n'.format(custom_split_name, custom_str[offset: offset + split_count]))
            offset += split_count
        cpp_file.write('const std::string {}::custom_model_{{\n    {}\n}};\n\n'.format(class_name, ' + '.join(custom_split_names)))

        cpp_file.write('const std::vector<int> {}::counts_{{\n    {}\n}};\n\n'.format(class_name, ', '.join(blob_counts)))
        cpp_file.write('const std::vector<const void *> {}::weights_{{\n    {}\n}};\n\n'.format(class_name, ',\n    '.join(blob_names)))
        cpp_file.write('const std::vector<std::string> {}::types_{{\n    "{}"\n}};\n\n'.format(class_name, '",\n    "'.join(blob_types)))

        cpp_file.write('}  // namespace Shadow\n')

    ########## write network proto definition to hpp ##########
    with open('{}/{}'.format(root, model_name_hpp), 'w') as hpp_file:
        hpp_file.write('#ifndef SHADOW_{}_HPP\n'.format(class_name) +
                       "#define SHADOW_{}_HPP\n\n".format(class_name))

        hpp_file.write('#include <cstring>\n' +
                       '#include <string>\n' +
                       '#include <vector>\n\n')

        hpp_file.write('namespace Shadow {\n\n')

        hpp_file.write('class {} {{\n'.format(class_name) +
                       ' public:\n')
        hpp_file.write('#if defined(USE_Protobuf)\n')
        hpp_file.write('  static const std::string &proto_model() { return proto_model_; }\n')
        hpp_file.write('#elif defined(USE_JSON)\n')
        hpp_file.write('  static const std::string &json_model() { return json_model_; }\n')
        hpp_file.write('#else\n')
        hpp_file.write('  static const std::string &custom_model() { return custom_model_; }\n')
        hpp_file.write('#endif\n\n')

        hpp_file.write('  static const std::vector<const void *> &weights() { return weights_; }\n')
        hpp_file.write('  static const std::vector<std::string> &types() { return types_; }\n')
        hpp_file.write('  static const std::vector<int> &counts() { return counts_; }\n\n')

        hpp_file.write('  static const void *weights(int n) { return weights_[n]; }\n')
        hpp_file.write('  static const std::string &types(int n) { return types_[n]; }\n')
        hpp_file.write('  static const int counts(int n) { return counts_[n]; }\n\n')

        hpp_file.write(' private:\n')
        hpp_file.write('  static const std::string proto_model_;\n')
        hpp_file.write('  static const std::string json_model_;\n')
        hpp_file.write('  static const std::string custom_model_;\n\n')

        hpp_file.write('  static const std::vector<const void *> weights_;\n')
        hpp_file.write('  static const std::vector<std::string> types_;\n')
        hpp_file.write('  static const std::vector<int> counts_;\n')
        hpp_file.write('};\n\n')

        hpp_file.write('}  // namespace Shadow\n\n')

        hpp_file.write('#endif  // SHADOW_{}_HPP\n'.format(class_name))

    ########## write extern weights definition to hpp ##########
    with open('{}/{}'.format(root, model_name_weights_hpp), 'w') as weights_file:
        weights_file.write('#ifndef SHADOW_{}_WEIGHTS_HPP\n'.format(class_name))
        weights_file.write('#define SHADOW_{}_WEIGHTS_HPP\n\n'.format(class_name))

        weights_file.write('namespace Shadow {\n\n')

        for blob_name, blob_type in zip(blob_names, blob_types):
            weights_file.write('extern const {} {}[];\n'.format(blob_type, blob_name))

        weights_file.write('\n}  // namespace Shadow\n\n')

        weights_file.write('#endif  // SHADOW_{}_WEIGHTS_HPP\n'.format(class_name))


def write_weights(net_param, root, model_name):
    weight_prefix = model_name.lower()
    model_name_weights_hpp = model_name + '_weights.hpp'

    for op_param in net_param.op:
        weight_blobs = get_blobs(net_param, op_param)

        if len(weight_blobs) == 0:
            continue

        op_name = find_replace(op_param.name, ['/', '-', ':'], '_')

        with open('{}/{}_{}.cpp'.format(root, model_name, op_name), 'w') as weights_file:
            weights_file.write('#include "{}"\n\n'.format(model_name_weights_hpp))

            weights_file.write('namespace Shadow {\n\n')

            for blob in weight_blobs:
                blob_name = find_replace(blob.name, ['/', '-', ':'], '_')
                blob_type = blob.type if blob.HasField('type') else 'float'

                weights_file.write('const {} {}_{}_[] = {{\n'.format(blob_type, weight_prefix, blob_name))

                if blob_type == 'float':
                    dump_weights(weights_file, blob.data_f, '{:f}')
                elif blob_type == 'int':
                    dump_weights(weights_file, blob.data_i, '{:d}')
                elif blob_type == 'unsigned char':
                    assert len(blob.data_b) == 1
                    dump_weights(weights_file, blob.data_b[0], '{:d}')
                else:
                    raise ValueError('Unknown blob type', blob_type)

                weights_file.write('\n};\n\n')

            weights_file.write('}  // namespace Shadow\n')


def write_proto_to_files(network, root, model_name):
    import os
    os.makedirs(root, exist_ok=True)
    num_net = len(network.meta_net_param.network)
    code_names = [model_name] if num_net == 1 else ['{}_net_{}'.format(model_name, n) for n in range(num_net)]
    for net_param, code_name in zip(network.meta_net_param.network, code_names):
        code_name = find_replace(code_name, ['.', '-'], '_')
        write_defines(net_param, root, code_name)
        write_weights(net_param, root, code_name)
