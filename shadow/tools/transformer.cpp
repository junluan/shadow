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

  std::vector<int> blob_counts;
  std::vector<std::string> blob_names, blob_types;
  shadow::NetParam net(shadow_net);
  for (int o = 0; o < net.op_size(); ++o) {
    auto op_param = net.mutable_op(o);
    if (op_param->blobs_size() == 0) continue;
    const auto& op_name = Util::find_replace(op_param->name(), "/", "_");
    int blob_count = 0;
    for (const auto& blob : op_param->blobs()) {
      const auto blob_type = blob.has_type() ? blob.type() : "float";
      if (blob_type == "float") {
        blob_counts.push_back(blob.data_f_size());
      } else if (blob_type == "int") {
        blob_counts.push_back(blob.data_i_size());
      } else if (blob_type == "unsigned char") {
        CHECK_EQ(blob.data_b_size(), 1);
        blob_counts.push_back(static_cast<int>(blob.data_b(0).size()));
      } else {
        LOG(FATAL) << "Unknown blob type " << blob_type;
      }
      std::stringstream ss;
      ss << weight_prefix << "_" << op_name << "_weight_" << blob_count++
         << "_";
      blob_names.push_back(ss.str());
      blob_types.push_back(blob_type);
    }
    for (int n = 0; n < op_param->blobs_size(); ++n) {
      op_param->mutable_blobs(n)->clear_data_f();
      op_param->mutable_blobs(n)->clear_data_i();
      op_param->mutable_blobs(n)->clear_data_b();
    }
  }

  std::string proto_str, json_str;
  IO::WriteProtoToText(net, &proto_str);
#if defined(SUPPORT_JSON)
  IO::WriteProtoToJsonText(net, &json_str, true);
#endif

  size_t split_count = 10000;
  auto proto_str_count = proto_str.size(), json_str_count = json_str.size();

  auto proto_split_off = proto_str_count % split_count;
  auto proto_split_num =
      proto_str_count / split_count + (proto_split_off > 0 ? 1 : 0);

  auto json_split_off = json_str_count % split_count;
  auto json_split_num =
      json_str_count / split_count + (json_split_off > 0 ? 1 : 0);

  std::vector<std::string> proto_split_names, json_split_names;
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
  cpp_file << "const std::string " << class_name << "::model_{\n"
           << Util::format_vector(proto_split_names, " + ", "    ")
           << "\n};\n\n";

#if defined(SUPPORT_JSON)
  offset = 0;
  for (int n = 0; n < json_split_num; ++n) {
    cpp_file << "const std::string " << json_split_names[n] << " = \nR\"(";
    cpp_file << json_str.substr(offset, split_count);
    cpp_file << ")\";\n\n";
    offset += split_count;
  }
  cpp_file << "const std::string " << class_name << "::json_model_{\n"
           << Util::format_vector(json_split_names, " + ", "    ")
           << "\n};\n\n";
#endif

  cpp_file << "const std::vector<int> " << class_name << "::counts_{\n"
           << Util::format_vector(blob_counts, ", ", "    ") << "\n};\n\n";

  cpp_file << "const std::vector<const void *> " << class_name
           << "::weights_{\n"
           << Util::format_vector(blob_names, ",\n    ", "    ") << "\n};\n\n";

  cpp_file << "const std::vector<std::string> " << class_name << "::types_{\n"
           << Util::format_vector(blob_types, "\",\n    \"", "    \"", "\"")
           << "\n};\n\n";

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

  hpp_file << "  static const std::vector<const void *> &weights() { "
              "return weights_; }\n";
  hpp_file << "  static const std::vector<std::string> &types() { return "
              "types_; }\n";
  hpp_file << "  static const std::vector<int> &counts() { return counts_; "
              "}\n\n";

  hpp_file << "  static const void *weights(int n) { return weights_[n]; }\n";
  hpp_file << "  static const std::string types(int n) { return types_[n]; }\n";
  hpp_file << "  static const int counts(int n) { return counts_[n]; }\n\n";

  hpp_file << "  static const int num() { return " << blob_counts.size()
           << "; }\n\n";

  hpp_file << " private:\n";
  hpp_file << "  static const std::string model_;\n";
#if defined(SUPPORT_JSON)
  hpp_file << "  static const std::string json_model_;\n";
#endif
  hpp_file << "\n";

  hpp_file << "  static const std::vector<const void *> weights_;\n";
  hpp_file << "  static const std::vector<std::string> types_;\n";
  hpp_file << "  static const std::vector<int> counts_;\n";
  hpp_file << "};\n\n";

  hpp_file << "#endif  // SHADOW_" << class_name << "_HPP\n";

  hpp_file.close();

  //########## write extern weights definition to hpp ##########//
  std::ofstream weight_file(root + "/" + model_name_weights_hpp);

  weight_file << "#ifndef SHADOW_" << class_name << "_WEIGHTS_HPP\n"
              << "#define SHADOW_" << class_name << "_WEIGHTS_HPP\n\n";

  for (int n = 0; n < blob_names.size(); ++n) {
    const auto &blob_name = blob_names[n], blob_type = blob_types[n];
    weight_file << "extern const " << blob_type << " " << blob_name << "[];\n";
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

  for (const auto& op_param : shadow_net.op()) {
    if (op_param.blobs_size() == 0) continue;
    const auto& op_name = Util::find_replace(op_param.name(), "/", "_");
    std::ofstream file(root + "/" + model_name + "_" + op_name + ".cpp");

    file << "#include \"" << model_name_weights_hpp << "\"\n\n";
    int blob_count = 0;
    for (const auto& blob : op_param.blobs()) {
      const auto blob_type = blob.has_type() ? blob.type() : "float";
      file << "const " << blob_type << " " << weight_prefix << "_" << op_name
           << "_weight_" << blob_count++ << "_[] = {\n";
      int data_size = 0;
      int count = 0, num_of_line = 10;
      if (blob_type == "float") {
        data_size = blob.data_f_size();
      } else if (blob_type == "int") {
        data_size = blob.data_i_size();
      } else if (blob_type == "unsigned char") {
        CHECK_EQ(blob.data_b_size(), 1);
        data_size = static_cast<int>(blob.data_b(0).size());
      } else {
        LOG(FATAL) << op_name << ": Failed to write blob " << blob_count
                   << "weights";
      }
      for (int i = 0; i < data_size; ++i) {
        if (count > 0) {
          file << ",";
        }
        if (count > 0 && count % num_of_line == 0) {
          file << "\n";
        }
        if (blob_type == "float") {
          file << blob.data_f(i);
        } else if (blob_type == "int") {
          file << blob.data_i(i);
        } else if (blob_type == "unsigned char") {
          auto uc_data_ptr = static_cast<unsigned char*>(
              static_cast<void*>(const_cast<char*>(blob.data_b(0).data())));
          file << static_cast<int>(uc_data_ptr[i]);
        }
        count++;
      }
      file << "};\n\n";
    }
    file.close();
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