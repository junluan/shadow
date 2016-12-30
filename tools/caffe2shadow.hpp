#ifndef SHADOW_TOOLS_CAFFE2SHADOW_HPP
#define SHADOW_TOOLS_CAFFE2SHADOW_HPP

#include "caffe.pb.h"
#include "shadow.pb.h"

#include "shadow/util/io.hpp"
#include "shadow/util/type.hpp"
#include "shadow/util/util.hpp"

namespace Caffe2Shadow {

void ConvertCommon(const caffe::NetParameter& caffe_model,
                   const caffe::LayerParameter& caffe_layer,
                   shadow::LayerParameter* shadow_layer) {
  const auto& layer_name = caffe_layer.name();
  shadow_layer->set_name(layer_name);
  for (const auto& top_name : caffe_layer.top()) {
    shadow_layer->add_top(top_name);
  }
  for (const auto& bottom_name : caffe_layer.bottom()) {
    shadow_layer->add_bottom(bottom_name);
  }
  for (const auto& layer : caffe_model.layer()) {
    if (!layer_name.compare(layer.name())) {
      for (const auto& caffe_blob : layer.blobs()) {
        auto shadow_blob = shadow_layer->add_blobs();
        for (const auto& dim : caffe_blob.shape().dim()) {
          shadow_blob->mutable_shape()->add_dim(dim);
        }
        for (const auto& data : caffe_blob.data()) {
          shadow_blob->add_data(data);
        }
      }
      break;
    }
  }
}

void ConvertData(const caffe::LayerParameter& data_model,
                 const caffe::LayerParameter& data_deploy,
                 shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_name(data_deploy.name());
  shadow_layer->set_type("Data");
  for (const auto& top_name : data_deploy.top()) {
    shadow_layer->add_top(top_name);
  }

  auto shadow_param = shadow_layer->mutable_data_param();
  if (data_model.has_transform_param()) {
    const auto& caffe_param = data_model.transform_param();
    if (caffe_param.has_scale()) {
      shadow_param->set_scale(caffe_param.scale());
    }
    if (caffe_param.mean_value_size() > 0) {
      for (const auto& mean_val : caffe_param.mean_value()) {
        shadow_param->add_mean_value(mean_val);
      }
    }
  }
}

void ConvertActivate(const caffe::NetParameter& caffe_model,
                     const caffe::LayerParameter& caffe_layer,
                     shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Activate");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_activate_param();
  if (!caffe_layer.type().compare("ReLU")) {
    shadow_param->set_type(shadow::ActivateParameter_ActivateType_Relu);
  }
}

void ConvertBatchNorm(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("BatchNorm");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_batch_norm_param();
  if (caffe_layer.has_batch_norm_param()) {
    const auto& caffe_param = caffe_layer.batch_norm_param();
    if (caffe_param.has_use_global_stats()) {
      shadow_param->set_use_global_stats(caffe_param.use_global_stats());
    }
  }
}

void ConvertBias(const caffe::NetParameter& caffe_model,
                 const caffe::LayerParameter& caffe_layer,
                 shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Bias");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_bias_param();
  if (caffe_layer.has_bias_param()) {
    const auto& caffe_param = caffe_layer.bias_param();
    if (caffe_param.has_axis()) {
      shadow_param->set_axis(caffe_param.axis());
    }
    if (caffe_param.has_num_axes()) {
      shadow_param->set_num_axes(caffe_param.num_axes());
    }
  }
}

void ConvertConcat(const caffe::NetParameter& caffe_model,
                   const caffe::LayerParameter& caffe_layer,
                   shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Concat");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_concat_param();
  if (caffe_layer.has_concat_param()) {
    const auto& caffe_param = caffe_layer.concat_param();
    if (caffe_param.has_axis()) {
      shadow_param->set_axis(caffe_param.axis());
    }
  }
}

void ConvertConnected(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Connected");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_connected_param();
  if (caffe_layer.has_inner_product_param()) {
    const auto& caffe_param = caffe_layer.inner_product_param();
    if (caffe_param.has_num_output()) {
      shadow_param->set_num_output(caffe_param.num_output());
    }
    if (caffe_param.has_bias_term() && !caffe_param.bias_term()) {
      shadow_param->set_bias_term(false);
    }
    if (caffe_param.has_transpose() && caffe_param.transpose()) {
      shadow_param->set_transpose(true);
    }
  }
}

void ConvertConv(const caffe::NetParameter& caffe_model,
                 const caffe::LayerParameter& caffe_layer,
                 shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Convolution");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_convolution_param();
  if (caffe_layer.has_convolution_param()) {
    const auto& caffe_param = caffe_layer.convolution_param();
    if (caffe_param.has_num_output()) {
      shadow_param->set_num_output(caffe_param.num_output());
    }
    if (caffe_param.kernel_size_size() > 0) {
      shadow_param->set_kernel_size(caffe_param.kernel_size(0));
    }
    if (caffe_param.stride_size() > 0) {
      shadow_param->set_stride(caffe_param.stride(0));
    }
    if (caffe_param.pad_size() > 0) {
      shadow_param->set_pad(caffe_param.pad(0));
    }
    if (caffe_param.dilation_size() > 0) {
      shadow_param->set_dilation(caffe_param.dilation(0));
    }
    if (caffe_param.has_bias_term() && !caffe_param.bias_term()) {
      shadow_param->set_bias_term(false);
    }
  }
}

void ConvertFlatten(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Flatten");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_flatten_param();
  if (caffe_layer.has_flatten_param()) {
    const auto& caffe_param = caffe_layer.flatten_param();
    if (caffe_param.has_axis()) {
      shadow_param->set_axis(caffe_param.axis());
    }
    if (caffe_param.has_end_axis()) {
      shadow_param->set_end_axis(caffe_param.end_axis());
    }
  }
}

void ConvertNormalize(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Normalize");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_normalize_param();
  if (caffe_layer.has_norm_param()) {
    const auto& caffe_param = caffe_layer.norm_param();
    if (caffe_param.has_across_spatial()) {
      shadow_param->set_across_spatial(caffe_param.across_spatial());
    }
    if (caffe_param.has_channel_shared()) {
      shadow_param->set_channel_shared(caffe_param.channel_shared());
    }
  }
}

void ConvertPermute(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Permute");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_permute_param();
  if (caffe_layer.has_permute_param()) {
    const auto& caffe_param = caffe_layer.permute_param();
    for (int i = 0; i < caffe_param.order_size(); ++i) {
      shadow_param->add_order(caffe_param.order(i));
    }
  }
}

void ConvertPooling(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Pooling");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_pooling_param();
  if (caffe_layer.has_pooling_param()) {
    const auto& caffe_param = caffe_layer.pooling_param();
    if (caffe_param.has_pool()) {
      if (caffe_param.pool() == caffe::PoolingParameter_PoolMethod_MAX) {
        shadow_param->set_pool(shadow::PoolingParameter_PoolType_Max);
      } else if (caffe_param.pool() == caffe::PoolingParameter_PoolMethod_AVE) {
        shadow_param->set_pool(shadow::PoolingParameter_PoolType_Ave);
      }
    }
    if (caffe_param.has_kernel_size()) {
      shadow_param->set_kernel_size(caffe_param.kernel_size());
    }
    if (caffe_param.has_stride()) {
      shadow_param->set_stride(caffe_param.stride());
    }
    if (caffe_param.has_pad()) {
      shadow_param->set_pad(caffe_param.pad());
    }
    if (caffe_param.has_global_pooling()) {
      shadow_param->set_global_pooling(caffe_param.global_pooling());
    }
  }
}

void ConvertPriorBox(const caffe::NetParameter& caffe_model,
                     const caffe::LayerParameter& caffe_layer,
                     shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("PriorBox");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_prior_box_param();
  if (caffe_layer.has_prior_box_param()) {
    const auto& caffe_param = caffe_layer.prior_box_param();
    if (caffe_param.has_min_size()) {
      shadow_param->set_min_size(caffe_param.min_size());
    }
    if (caffe_param.has_max_size()) {
      shadow_param->set_max_size(caffe_param.max_size());
    }
    for (const auto& ratio : caffe_param.aspect_ratio()) {
      shadow_param->add_aspect_ratio(ratio);
    }
    if (caffe_param.has_flip()) {
      shadow_param->set_flip(caffe_param.flip());
    }
    if (caffe_param.has_clip()) {
      shadow_param->set_clip(caffe_param.clip());
    }
    for (const auto& variance : caffe_param.variance()) {
      shadow_param->add_variance(variance);
    }
  }
}

void ConvertReshape(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Reshape");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_reshape_param();
  if (caffe_layer.has_reshape_param()) {
    const auto& caffe_param = caffe_layer.reshape_param();
    for (const auto& dim : caffe_param.shape().dim()) {
      shadow_param->mutable_shape()->add_dim(dim);
    }
    if (caffe_param.has_axis() && caffe_param.axis() != 0) {
      shadow_param->set_axis(caffe_param.axis());
    }
    if (caffe_param.has_num_axes() && caffe_param.num_axes() != -1) {
      shadow_param->set_num_axes(caffe_param.num_axes());
    }
  }
}

void ConvertScale(const caffe::NetParameter& caffe_model,
                  const caffe::LayerParameter& caffe_layer,
                  shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Scale");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_scale_param();
  if (caffe_layer.has_scale_param()) {
    const auto& caffe_param = caffe_layer.scale_param();
    if (caffe_param.has_axis()) {
      shadow_param->set_axis(caffe_param.axis());
    }
    if (caffe_param.has_num_axes()) {
      shadow_param->set_num_axes(caffe_param.num_axes());
    }
    if (caffe_param.has_bias_term()) {
      shadow_param->set_bias_term(caffe_param.bias_term());
    }
  }
}

void ConvertSoftmax(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Softmax");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  auto shadow_param = shadow_layer->mutable_softmax_param();
  if (caffe_layer.has_softmax_param()) {
    const auto& caffe_param = caffe_layer.softmax_param();
    if (caffe_param.has_axis() && caffe_param.axis() != 1) {
      shadow_param->set_axis(caffe_param.axis());
    }
  }
}

void Convert(const caffe::NetParameter& caffe_deploy,
             const caffe::NetParameter& caffe_model,
             shadow::NetParameter* shadow_net) {
  shadow_net->set_name(caffe_deploy.name());
  ConvertData(caffe_model.layer(0), caffe_deploy.layer(0), shadow_net);
  for (int l = 1; l < caffe_deploy.layer_size(); ++l) {
    const auto& caffe_layer = caffe_deploy.layer(l);
    const auto& layer_type = caffe_layer.type();
    if (!layer_type.compare("ReLU")) {
      ConvertActivate(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("BatchNorm")) {
      ConvertBatchNorm(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Bias")) {
      ConvertBias(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Concat")) {
      ConvertConcat(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("InnerProduct")) {
      ConvertConnected(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Convolution")) {
      ConvertConv(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Flatten")) {
      ConvertFlatten(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Normalize")) {
      ConvertNormalize(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Permute")) {
      ConvertPermute(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Pooling")) {
      ConvertPooling(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("PriorBox")) {
      ConvertPriorBox(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Reshape")) {
      ConvertReshape(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Scale")) {
      ConvertScale(caffe_model, caffe_layer, shadow_net);
    } else if (!layer_type.compare("Softmax")) {
      ConvertSoftmax(caffe_model, caffe_layer, shadow_net);
    } else {
      LOG(WARNING) << "Layer type: " << layer_type << " is not recognized!";
    }
  }
}

}  // namespace Caffe2Shadow

namespace Caffe2Shadow {

using google::protobuf::Message;

void WriteDefines(const shadow::NetParameter& shadow_net,
                  const std::string& root, const std::string& model_name) {
  // write network proto definition to cpp
  shadow::NetParameter net(shadow_net);
  for (int l = 0; l < net.layer_size(); ++l) {
    auto layer_param = net.mutable_layer(l);
    for (int n = 0; n < layer_param->blobs_size(); ++n) {
      layer_param->mutable_blobs(n)->clear_data();
    }
  }

  std::ofstream cpp_file(root + "/" + model_name + ".cpp");
  std::string proto_str;
  IO::WriteProtoToText(net, &proto_str);

  size_t split_count = 10000, str_count = proto_str.size();
  size_t split_off = str_count % split_count;
  size_t split_num = str_count / split_count + (split_off > 0 ? 1 : 0);

  cpp_file << "#include \"" << model_name << ".hpp\"\n\n";

  size_t offset = 0;
  for (int n = 0; n < split_num; ++n) {
    cpp_file << "const std::string Model::model_" << n << "_ = \nR\"(";
    cpp_file << proto_str.substr(offset, split_count);
    cpp_file << ")\";\n\n";
    offset += split_count;
  }

  cpp_file.close();

  // write all definitions to hpp
  VecInt weight_counts;
  VecString weight_names;
  for (const auto& layer_param : shadow_net.layer()) {
    if (layer_param.blobs_size() == 0) continue;
    int count = 0;
    for (const auto& blob : layer_param.blobs()) {
      count += blob.data_size();
    }
    weight_counts.push_back(count);
    weight_names.push_back(model_name + "_" + layer_param.name());
  }

  std::ofstream file(root + "/" + model_name + ".hpp");

  file << "#ifndef SHADOW_MODEL_HPP\n"
          "#define SHADOW_MODEL_HPP\n\n";

  file << "#include <cstring>\n"
       << "#include <string>\n"
       << "#include <vector>\n\n";

  file << "static int counts_[] = "
       << Util::format_vector(weight_counts, ", ", "{", "}") << ";\n\n";

  file << "class Model {\n"
          " public:\n"
          "  static const std::string model() { return";

  for (int i = 0; i < split_num - 1; ++i) {
    file << " model_" << i << "_ +";
  }
  file << " model_" << split_num - 1 << "_; }\n\n";

  file << "  static const float *weight(int n) {\n"
          "    switch (n) {\n";
  for (int i = 0; i < weight_names.size(); ++i) {
    file << "      case " << i << ":\n"
         << "        return " << weight_names[i] << "_;\n";
  }
  file << "      default:\n"
          "        return nullptr;\n"
          "    }\n"
          "  }\n";
  file << "  static const std::vector<const float *> get_weights() {\n"
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

  file << "  static const int count(int n) { return counts_[n]; }\n"
          "  static const int count() {\n"
          "    int cou = 0;\n"
          "    for (int n = 0; n < num(); ++n) cou += count(n);\n"
          "    return cou;\n"
          "  }\n\n";

  file << "  static const int num() { return " << weight_names.size()
       << "; }\n\n";

  file << " private:\n";

  for (int i = 0; i < split_num; ++i) {
    file << "  static const std::string model_" << i << "_;\n";
  }
  file << "\n";
  for (const auto& weight_name : weight_names) {
    file << "  static const float *" << weight_name << "_;\n";
  }
  file << "};\n\n";

  file << "#endif  // SHADOW_MODEL_HPP\n";

  file.close();
}

void WriteWeights(const shadow::NetParameter& shadow_net,
                  const std::string& root, const std::string& model_name) {
  for (const auto& layer_param : shadow_net.layer()) {
    if (layer_param.blobs_size() == 0) continue;

    std::string weight_name = model_name + "_" + layer_param.name();
    std::ofstream file(root + "/" + weight_name + ".cpp");

    file << "#include \"" << model_name << ".hpp\"\n\n";
    file << "const float " << weight_name << "_weights[] = {\n";

    int count = 0, num_of_line = 10;
    for (const auto& blob : layer_param.blobs()) {
      for (const auto& data : blob.data()) {
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

    file << "};\n\n";
    file << "const float *Model::" << weight_name << "_ = " << weight_name
         << "_weights;\n";

    file.close();
  }
}

void WriteProtoToFiles(const shadow::NetParameter& shadow_net,
                       const std::string& root, const std::string& model_name) {
  WriteDefines(shadow_net, root, model_name);
  WriteWeights(shadow_net, root, model_name);
}

void WriteProtoToBinary(const Message& proto, const std::string& root,
                        const std::string& model_name) {
  std::string filename = root + "/" + model_name + ".shadowmodel";
  IO::WriteProtoToBinaryFile(proto, filename);
}

}  // namespace Caffe2Shadow

#endif  // SHADOW_TOOLS_CAFFE2SHADOW_HPP
