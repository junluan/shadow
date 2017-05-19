#ifndef SHADOW_TOOLS_CAFFE2SHADOW_HPP
#define SHADOW_TOOLS_CAFFE2SHADOW_HPP

#include "proto/caffe.pb.h"
#include "proto/shadow.pb.h"

#include "util/io.hpp"
#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

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
        if (caffe_blob.shape().dim_size() > 0) {
          for (const auto& dim : caffe_blob.shape().dim()) {
            shadow_blob->mutable_shape()->add_dim(dim);
          }
        } else {
          shadow_blob->mutable_shape()->add_dim(caffe_blob.data_size());
        }
        for (const auto& data : caffe_blob.data()) {
          shadow_blob->add_data(data);
        }
      }
      break;
    }
  }
}

void ConvertData(const caffe::LayerParameter& data_layer,
                 const std::vector<int>& input_shape,
                 shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_name(data_layer.name());
  shadow_layer->set_type("Data");
  for (const auto& top_name : data_layer.top()) {
    shadow_layer->add_top(top_name);
  }

  if (data_layer.has_input_param()) {
    auto shadow_param = shadow_layer->mutable_data_param();
    const auto& caffe_param = data_layer.input_param();
    for (const auto& dim : caffe_param.shape(0).dim()) {
      shadow_param->mutable_data_shape()->add_dim(dim);
    }
  } else {
    auto shadow_param = shadow_layer->mutable_data_param();
    for (const auto& dim : input_shape) {
      shadow_param->mutable_data_shape()->add_dim(dim);
    }
  }

  if (data_layer.has_transform_param()) {
    auto shadow_param = shadow_layer->mutable_data_param();
    const auto& caffe_param = data_layer.transform_param();
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
  } else if (!caffe_layer.type().compare("PReLU")) {
    if (caffe_layer.has_prelu_param()) {
      const auto& caffe_param = caffe_layer.prelu_param();
      if (caffe_param.has_channel_shared()) {
        shadow_param->set_channel_shared(caffe_param.channel_shared());
      }
      shadow_param->set_type(shadow::ActivateParameter_ActivateType_PRelu);
    }
  }
}

void ConvertBatchNorm(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("BatchNorm");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  if (caffe_layer.has_batch_norm_param()) {
    auto shadow_param = shadow_layer->mutable_batch_norm_param();
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

  if (caffe_layer.has_bias_param()) {
    auto shadow_param = shadow_layer->mutable_bias_param();
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

  if (caffe_layer.has_concat_param()) {
    auto shadow_param = shadow_layer->mutable_concat_param();
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

  if (caffe_layer.has_inner_product_param()) {
    auto shadow_param = shadow_layer->mutable_connected_param();
    const auto& caffe_param = caffe_layer.inner_product_param();
    if (caffe_param.has_num_output()) {
      shadow_param->set_num_output(caffe_param.num_output());
    }
    if (caffe_param.has_bias_term()) {
      shadow_param->set_bias_term(caffe_param.bias_term());
    }
    if (caffe_param.has_transpose()) {
      shadow_param->set_transpose(caffe_param.transpose());
    }
  }
}

void ConvertConv(const caffe::NetParameter& caffe_model,
                 const caffe::LayerParameter& caffe_layer,
                 shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Convolution");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  if (caffe_layer.has_convolution_param()) {
    auto shadow_param = shadow_layer->mutable_convolution_param();
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
    if (caffe_param.has_group() && caffe_param.group() != 1) {
      shadow_param->set_group(caffe_param.group());
    }
  }
}

void ConvertFlatten(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Flatten");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  if (caffe_layer.has_flatten_param()) {
    auto shadow_param = shadow_layer->mutable_flatten_param();
    const auto& caffe_param = caffe_layer.flatten_param();
    if (caffe_param.has_axis()) {
      shadow_param->set_axis(caffe_param.axis());
    }
    if (caffe_param.has_end_axis()) {
      shadow_param->set_end_axis(caffe_param.end_axis());
    }
  }
}

void ConvertLRN(const caffe::NetParameter& caffe_model,
                const caffe::LayerParameter& caffe_layer,
                shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("LRN");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  if (caffe_layer.has_lrn_param()) {
    auto shadow_param = shadow_layer->mutable_lrn_param();
    const auto& caffe_param = caffe_layer.lrn_param();
    if (caffe_param.has_local_size()) {
      shadow_param->set_local_size(caffe_param.local_size());
    }
    if (caffe_param.has_alpha()) {
      shadow_param->set_alpha(caffe_param.alpha());
    }
    if (caffe_param.has_beta()) {
      shadow_param->set_beta(caffe_param.beta());
    }
    if (caffe_param.has_norm_region()) {
      if (caffe_param.norm_region() ==
          caffe::LRNParameter_NormRegion_ACROSS_CHANNELS) {
        shadow_param->set_norm_region(
            shadow::LRNParameter_NormRegion_AcrossChannels);
      } else {
        LOG(FATAL) << "Unsupported norm region method: Within Channel!";
      }
    }
    if (caffe_param.has_k()) {
      shadow_param->set_k(caffe_param.k());
    }
  }
}

void ConvertNormalize(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Normalize");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  if (caffe_layer.has_norm_param()) {
    auto shadow_param = shadow_layer->mutable_normalize_param();
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

  if (caffe_layer.has_permute_param()) {
    auto shadow_param = shadow_layer->mutable_permute_param();
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

  if (caffe_layer.has_pooling_param()) {
    auto shadow_param = shadow_layer->mutable_pooling_param();
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

  if (caffe_layer.has_prior_box_param()) {
    auto shadow_param = shadow_layer->mutable_prior_box_param();
    const auto& caffe_param = caffe_layer.prior_box_param();
    for (const auto& min_size : caffe_param.min_size()) {
      shadow_param->add_min_size(min_size);
    }
    for (const auto& max_size : caffe_param.max_size()) {
      shadow_param->add_max_size(max_size);
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
    if (caffe_param.has_step()) {
      shadow_param->set_step(caffe_param.step());
    }
    if (caffe_param.has_offset()) {
      shadow_param->set_offset(caffe_param.offset());
    }
  }
}

void ConvertReshape(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  auto shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Reshape");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  if (caffe_layer.has_reshape_param()) {
    auto shadow_param = shadow_layer->mutable_reshape_param();
    const auto& caffe_param = caffe_layer.reshape_param();
    for (const auto& dim : caffe_param.shape().dim()) {
      shadow_param->mutable_shape()->add_dim(dim);
    }
    if (caffe_param.has_axis()) {
      shadow_param->set_axis(caffe_param.axis());
    }
    if (caffe_param.has_num_axes()) {
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

  if (caffe_layer.has_scale_param()) {
    auto shadow_param = shadow_layer->mutable_scale_param();
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

  if (caffe_layer.has_softmax_param()) {
    auto shadow_param = shadow_layer->mutable_softmax_param();
    const auto& caffe_param = caffe_layer.softmax_param();
    if (caffe_param.has_axis()) {
      shadow_param->set_axis(caffe_param.axis());
    }
  }
}

void Convert(const caffe::NetParameter& caffe_deploy,
             const caffe::NetParameter& caffe_model,
             const std::vector<int>& input_shape,
             shadow::NetParameter* shadow_net) {
  shadow_net->set_name(caffe_deploy.name());
  ConvertData(caffe_deploy.layer(0), input_shape, shadow_net);
  for (int l = 1; l < caffe_deploy.layer_size(); ++l) {
    const auto& caffe_layer = caffe_deploy.layer(l);
    const auto& layer_type = caffe_layer.type();
    if (!layer_type.compare("ReLU") || !layer_type.compare("PReLU")) {
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
    } else if (!layer_type.compare("LRN")) {
      ConvertLRN(caffe_model, caffe_layer, shadow_net);
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

void WriteDefines(const shadow::NetParameter& shadow_net,
                  const std::string& root, const std::string& model_name) {
  auto class_name = model_name;
  std::transform(model_name.begin(), model_name.end(), class_name.begin(),
                 ::toupper);
  const auto& model_name_cpp = model_name + ".cpp";
  const auto& model_name_hpp = model_name + ".hpp";
  const auto& model_name_weights_hpp = model_name + "_weights.hpp";

  int whole_count = 0;
  std::vector<int> weight_counts;
  shadow::NetParameter net(shadow_net);
  for (int l = 0; l < net.layer_size(); ++l) {
    auto layer_param = net.mutable_layer(l);
    if (layer_param->blobs_size() == 0) continue;
    int count = 0;
    for (const auto& blob : layer_param->blobs()) {
      count += blob.data_size();
    }
    whole_count += count;
    weight_counts.push_back(count);
    for (int n = 0; n < layer_param->blobs_size(); ++n) {
      layer_param->mutable_blobs(n)->clear_data();
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

  //########## write network proto definition to cpp ##########//
  std::ofstream cpp_file(root + "/" + model_name_cpp);

  cpp_file << "#include \"" << model_name_hpp << "\"\n"
           << "#include \"" << model_name_weights_hpp << "\"\n\n";

  size_t offset = 0;
  for (int n = 0; n < proto_split_num; ++n) {
    cpp_file << "const std::string model_" << n << "_ = \nR\"(";
    cpp_file << proto_str.substr(offset, split_count);
    cpp_file << ")\";\n\n";
    offset += split_count;
  }
  cpp_file << "const std::string " << class_name << "::model_ = ";
  for (int n = 0; n < proto_split_num - 1; ++n) {
    cpp_file << "model_" << n << "_ + ";
  }
  cpp_file << "model_" << proto_split_num - 1 << "_;\n\n";

#if defined(SUPPORT_JSON)
  offset = 0;
  for (int n = 0; n < json_split_num; ++n) {
    cpp_file << "const std::string json_model_" << n << "_ = \nR\"(";
    cpp_file << json_str.substr(offset, split_count);
    cpp_file << ")\";\n\n";
    offset += split_count;
  }
  cpp_file << "const std::string " << class_name << "::json_model_ = ";
  for (int n = 0; n < json_split_num - 1; ++n) {
    cpp_file << "json_model_" << n << "_ + ";
  }
  cpp_file << "json_model_" << json_split_num - 1 << "_;\n\n";
#endif

  cpp_file << "const std::vector<int> " << class_name << "::counts_"
           << Util::format_vector(weight_counts, ", ", "{", "}") << ";\n\n";

  cpp_file << "const std::vector<const float *> " << class_name
           << "::weights_{";
  for (int i = 0; i < weight_counts.size() - 1; ++i) {
    cpp_file << "weight_" << i << "_, ";
  }
  cpp_file << "weight_" << weight_counts.size() - 1 << "_};\n";

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

  for (int i = 0; i < weight_counts.size(); ++i) {
    weight_file << "extern const float weight_" << i << "_[];\n";
  }
  weight_file << "\n";

  weight_file << "#endif  // SHADOW_" << class_name << "_WEIGHTS_HPP\n";

  weight_file.close();
}

void WriteWeights(const shadow::NetParameter& shadow_net,
                  const std::string& root, const std::string& model_name) {
  const auto& model_name_weights_hpp = model_name + "_weights.hpp";

  int weight_count = 0;
  for (const auto& layer_param : shadow_net.layer()) {
    if (layer_param.blobs_size() == 0) continue;

    const auto& weight_name = Util::find_replace(layer_param.name(), "/", "_");
    std::ofstream file(root + "/" + model_name + "_" + weight_name + ".cpp");

    file << "#include \"" << model_name_weights_hpp << "\"\n\n";

    file << "const float weight_" << weight_count << "_[] = {\n";
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
    file << "};\n";

    file.close();

    weight_count++;
  }
}

void WriteProtoToFiles(const shadow::NetParameter& shadow_net,
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

}  // namespace Caffe2Shadow

}  // namespace Shadow

#endif  // SHADOW_TOOLS_CAFFE2SHADOW_HPP
