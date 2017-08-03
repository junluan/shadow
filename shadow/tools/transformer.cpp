#include "transformer.hpp"

namespace Shadow {

void WriteDefines(const shadow::NetParam& shadow_net, const std::string& root,
                  const std::string& model_name) {
  auto class_name = model_name;
  auto weight_prefix = model_name;
  std::transform(model_name.begin(), model_name.end(), class_name.begin(),
                 ::toupper);
  std::transform(model_name.begin(), model_name.end(), weight_prefix.begin(),
                 ::tolower);
  const auto& model_name_cpp = model_name + ".cpp";
  const auto& model_name_hpp = model_name + ".hpp";
  const auto& model_name_weights_hpp = model_name + "_weights.hpp";

  int whole_count = 0;
  std::vector<int> weight_counts;
  shadow::NetParam net(shadow_net);
  for (int o = 0; o < net.op_size(); ++o) {
    auto op_param = net.mutable_op(o);
    if (op_param->blobs_size() == 0) continue;
    int count = 0;
    for (const auto& blob : op_param->blobs()) {
      count += blob.data_f_size();
    }
    whole_count += count;
    weight_counts.push_back(count);
    for (int n = 0; n < op_param->blobs_size(); ++n) {
      op_param->mutable_blobs(n)->clear_data_f();
    }
  }

  std::string proto_str, json_str;
  IO::WriteProtoToText(net, &proto_str);
#if defined(SUPPORT_JSON)
  IO::WriteProtoToJsonText(net, &json_str);
#endif

  size_t split_count = 10000;
  auto proto_str_count = proto_str.size(), json_str_count = json_str.size();

  auto proto_split_off = proto_str_count % split_count;
  auto proto_split_num =
      proto_str_count / split_count + (proto_split_off > 0 ? 1 : 0);

  auto json_split_off = json_str_count % split_count;
  auto json_split_num =
      json_str_count / split_count + (json_split_off > 0 ? 1 : 0);

  std::vector<std::string> proto_split_names, json_split_names, weight_names;
  for (int n = 0; n < proto_split_num; ++n) {
    std::stringstream ss;
    ss << "model_" << n << "_";
    proto_split_names.push_back(ss.str());
  }
  for (int n = 0; n < json_split_num; ++n) {
    std::stringstream ss;
    ss << "json_model_" << n << "_";
    json_split_names.push_back(ss.str());
  }
  for (int n = 0; n < weight_counts.size(); ++n) {
    std::stringstream ss;
    ss << weight_prefix << "_weight_" << n << "_";
    weight_names.push_back(ss.str());
  }

  //########## write network proto definition to cpp ##########//
  std::ofstream cpp_file(root + "/" + model_name_cpp);

  cpp_file << "#include \"" << model_name_hpp << "\"\n"
           << "#include \"" << model_name_weights_hpp << "\"\n\n";

  size_t offset = 0;
  for (int n = 0; n < proto_split_num; ++n) {
    cpp_file << "const std::string " << proto_split_names[n] << " = \nR\"(";
    cpp_file << proto_str.substr(offset, split_count);
    cpp_file << ")\";\n\n";
    offset += split_count;
  }
  cpp_file << "const std::string " << class_name
           << "::model_ = " << Util::format_vector(proto_split_names, " + ")
           << ";\n\n";

#if defined(SUPPORT_JSON)
  offset = 0;
  for (int n = 0; n < json_split_num; ++n) {
    cpp_file << "const std::string " << json_split_names[n] << " = \nR\"(";
    cpp_file << json_str.substr(offset, split_count);
    cpp_file << ")\";\n\n";
    offset += split_count;
  }
  cpp_file << "const std::string " << class_name
           << "::json_model_ = " << Util::format_vector(json_split_names, " + ")
           << ";\n\n";
#endif

  cpp_file << "const std::vector<int> " << class_name << "::counts_{"
           << Util::format_vector(weight_counts, ", ") << "};\n\n";

  cpp_file << "const std::vector<const float *> " << class_name << "::weights_{"
           << Util::format_vector(weight_names, ", ") << "};\n\n";

  cpp_file.close();

  //########## write network proto definition to hpp ##########//
  std::ofstream hpp_file(root + "/" + model_name_hpp);

  hpp_file << "#ifndef SHADOW_" << class_name << "_HPP\n"
           << "#define SHADOW_" << class_name << "_HPP\n\n";

  hpp_file << "#include <cstring>\n"
           << "#include <string>\n"
           << "#include <vector>\n\n";

  hpp_file << "class " << class_name << " {\n"
           << " public:\n";
  hpp_file << "  static const std::string model() { return model_; }\n";
#if defined(SUPPORT_JSON)
  hpp_file
      << "  static const std::string json_model() { return json_model_; }\n";
#endif
  hpp_file << "\n";

  hpp_file
      << "  static const std::vector<const float *> get_weights() {\n"
         "    std::vector<const float *> weights;\n"
         "    for (int n = 0; n < num(); ++n) {\n"
         "      weights.push_back(weight(n));\n"
         "    }\n"
         "    return weights;\n"
         "  }\n"
         "  static void get_weights(float *weights_data) {\n"
         "    for (int n = 0; n < num(); ++n) {\n"
         "      memcpy(weights_data, weight(n), count(n) * sizeof(float));\n"
         "      weights_data += count(n);\n"
         "    }\n"
         "  }\n\n";

  hpp_file << "  static const int count() { return " << whole_count
           << "; }\n\n";

  hpp_file << " private:\n";
  hpp_file << "  static const int num() { return " << weight_counts.size()
           << "; }\n";
  hpp_file << "  static const int count(int n) { return counts_[n]; }\n";
  hpp_file << "  static const float *weight(int n) { return weights_[n]; }\n\n";

  hpp_file << "  static const std::string model_;\n";
#if defined(SUPPORT_JSON)
  hpp_file << "  static const std::string json_model_;\n";
#endif
  hpp_file << "\n";

  hpp_file << "  static const std::vector<int> counts_;\n";
  hpp_file << "  static const std::vector<const float *> weights_;\n";
  hpp_file << "};\n\n";

  hpp_file << "#endif  // SHADOW_" << class_name << "_HPP\n";

  hpp_file.close();

  //########## write extern weights definition to hpp ##########//
  std::ofstream weight_file(root + "/" + model_name_weights_hpp);

  weight_file << "#ifndef SHADOW_" << class_name << "_WEIGHTS_HPP\n"
              << "#define SHADOW_" << class_name << "_WEIGHTS_HPP\n\n";

  for (const auto& weight_name : weight_names) {
    weight_file << "extern const float " << weight_name << "[];\n";
  }
  weight_file << "\n";

  weight_file << "#endif  // SHADOW_" << class_name << "_WEIGHTS_HPP\n";

  weight_file.close();
}

void WriteWeights(const shadow::NetParam& shadow_net, const std::string& root,
                  const std::string& model_name) {
  auto weight_prefix = model_name;
  std::transform(model_name.begin(), model_name.end(), weight_prefix.begin(),
                 ::tolower);
  const auto& model_name_weights_hpp = model_name + "_weights.hpp";

  int weight_count = 0;
  for (const auto& op_param : shadow_net.op()) {
    if (op_param.blobs_size() == 0) continue;

    const auto& weight_name = Util::find_replace(op_param.name(), "/", "_");
    std::ofstream file(root + "/" + model_name + "_" + weight_name + ".cpp");

    file << "#include \"" << model_name_weights_hpp << "\"\n\n";

    file << "const float " << weight_prefix << "_weight_" << weight_count
         << "_[] = {\n";
    int count = 0, num_of_line = 10;
    for (const auto& blob : op_param.blobs()) {
      for (const auto& data : blob.data_f()) {
        if (count > 0) {
          file << ",";
        }
        if (count > 0 && count % num_of_line == 0) {
          file << "\n";
        }
        file << data;
        count++;
      }
    }
    file << "};\n";

    file.close();

    weight_count++;
  }
}

void WriteProtoToFiles(const shadow::NetParam& shadow_net,
                       const std::string& root, const std::string& model_name) {
  Util::make_directory(root);
  WriteDefines(shadow_net, root, model_name);
  WriteWeights(shadow_net, root, model_name);
}

void WriteProtoToBinary(const IO::Message& proto, const std::string& root,
                        const std::string& model_name) {
  Util::make_directory(root);
  std::string filename = root + "/" + model_name + ".shadowmodel";
  IO::WriteProtoToBinaryFile(proto, filename);
}

}  // namespace Shadow