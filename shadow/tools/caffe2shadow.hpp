#ifndef SHADOW_TOOLS_CAFFE2SHADOW_HPP
#define SHADOW_TOOLS_CAFFE2SHADOW_HPP

#include "proto/caffe.pb.h"
#include "proto/shadow.pb.h"

#include "util/io.hpp"
#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

namespace Caffe2Shadow {

#define INSTANTIATE_SET_SINGLE_ARGUMENT(T, fieldname)                    \
  static void set_##fieldname(shadow::OpParam* op_param,                 \
                              const std::string& name, const T& value) { \
    auto shadow_arg = op_param->add_arg();                               \
    shadow_arg->set_name(name);                                          \
    shadow_arg->set_##fieldname(value);                                  \
  }

INSTANTIATE_SET_SINGLE_ARGUMENT(float, s_f);
INSTANTIATE_SET_SINGLE_ARGUMENT(int, s_i);
INSTANTIATE_SET_SINGLE_ARGUMENT(unsigned int, s_i);
INSTANTIATE_SET_SINGLE_ARGUMENT(bool, s_i);
INSTANTIATE_SET_SINGLE_ARGUMENT(std::string, s_s);
#undef INSTANTIATE_SET_SINGLE_ARGUMENT

#define INSTANTIATE_SET_REPEATED_ARGUMENT(T, fieldname)      \
  static void set_##fieldname(shadow::OpParam* op_param,     \
                              const std::string& name,       \
                              const std::vector<T>& value) { \
    auto shadow_arg = op_param->add_arg();                   \
    shadow_arg->set_name(name);                              \
    for (const auto v : value) {                             \
      shadow_arg->add_##fieldname(v);                        \
    }                                                        \
  }

INSTANTIATE_SET_REPEATED_ARGUMENT(float, v_f);
INSTANTIATE_SET_REPEATED_ARGUMENT(int, v_i);
INSTANTIATE_SET_REPEATED_ARGUMENT(unsigned int, v_i);
INSTANTIATE_SET_REPEATED_ARGUMENT(bool, v_i);
INSTANTIATE_SET_REPEATED_ARGUMENT(std::string, v_s);
#undef INSTANTIATE_SET_REPEATED_ARGUMENT

void ConvertCommon(const caffe::NetParameter& caffe_model,
                   const caffe::LayerParameter& caffe_layer,
                   shadow::OpParam* shadow_op) {
  const auto& layer_name = caffe_layer.name();
  shadow_op->set_name(layer_name);
  for (const auto& top_name : caffe_layer.top()) {
    shadow_op->add_top(top_name);
  }
  for (const auto& bottom_name : caffe_layer.bottom()) {
    shadow_op->add_bottom(bottom_name);
  }
  for (const auto& layer : caffe_model.layer()) {
    if (!layer_name.compare(layer.name())) {
      for (const auto& caffe_blob : layer.blobs()) {
        auto shadow_blob = shadow_op->add_blobs();
        if (caffe_blob.shape().dim_size() > 0) {
          for (const auto dim : caffe_blob.shape().dim()) {
            shadow_blob->add_shape(dim);
          }
        } else {
          shadow_blob->add_shape(caffe_blob.data_size());
        }
        for (const auto data : caffe_blob.data()) {
          shadow_blob->add_data_f(data);
        }
      }
      break;
    }
  }
}

void ConvertInput(const caffe::NetParameter& caffe_deploy,
                  const std::vector<int>& input_shape,
                  shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_name("data");
  shadow_op->set_type("Data");
  for (const auto& top_name : caffe_deploy.input()) {
    shadow_op->add_top(top_name);
  }
  set_v_i(shadow_op, "data_shape", input_shape);
}

void ConvertData(const caffe::LayerParameter& data_layer,
                 const std::vector<int>& input_shape,
                 shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_name(data_layer.name());
  shadow_op->set_type("Data");
  for (const auto& top_name : data_layer.top()) {
    shadow_op->add_top(top_name);
  }

  if (data_layer.has_input_param()) {
    std::vector<int> shape;
    const auto& caffe_param = data_layer.input_param();
    for (const auto dim : caffe_param.shape(0).dim()) {
      shape.push_back(dim);
    }
    set_v_i(shadow_op, "data_shape", shape);
  } else {
    set_v_i(shadow_op, "data_shape", input_shape);
  }

  if (data_layer.has_transform_param()) {
    const auto& caffe_param = data_layer.transform_param();
    if (caffe_param.has_scale()) {
      set_s_f(shadow_op, "scale", caffe_param.scale());
    }
    if (caffe_param.mean_value_size() > 0) {
      std::vector<float> mean_value;
      for (const auto mean_val : caffe_param.mean_value()) {
        mean_value.push_back(mean_val);
      }
      set_v_f(shadow_op, "mean_value", mean_value);
    }
  }
}

void ConvertActivate(const caffe::NetParameter& caffe_model,
                     const caffe::LayerParameter& caffe_layer,
                     shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Activate");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  // Linear: 0, Relu: 1, Leaky: 2, PRelu: 3
  if (!caffe_layer.type().compare("ReLU")) {
    set_s_i(shadow_op, "type", 1);
  } else if (!caffe_layer.type().compare("PReLU")) {
    set_s_i(shadow_op, "type", 3);
    if (caffe_layer.has_prelu_param()) {
      const auto& caffe_param = caffe_layer.prelu_param();
      if (caffe_param.has_channel_shared()) {
        set_s_i(shadow_op, "channel_shared", caffe_param.channel_shared());
      }
    }
  }
}

void ConvertBatchNorm(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("BatchNorm");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_batch_norm_param()) {
    const auto& caffe_param = caffe_layer.batch_norm_param();
    if (caffe_param.has_use_global_stats()) {
      set_s_i(shadow_op, "use_global_stats", caffe_param.use_global_stats());
    }
  }
}

void ConvertBias(const caffe::NetParameter& caffe_model,
                 const caffe::LayerParameter& caffe_layer,
                 shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Bias");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_bias_param()) {
    const auto& caffe_param = caffe_layer.bias_param();
    if (caffe_param.has_axis()) {
      set_s_i(shadow_op, "axis", caffe_param.axis());
    }
    if (caffe_param.has_num_axes()) {
      set_s_i(shadow_op, "num_axes", caffe_param.num_axes());
    }
  }
}

void ConvertConcat(const caffe::NetParameter& caffe_model,
                   const caffe::LayerParameter& caffe_layer,
                   shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Concat");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_concat_param()) {
    const auto& caffe_param = caffe_layer.concat_param();
    if (caffe_param.has_axis()) {
      set_s_i(shadow_op, "axis", caffe_param.axis());
    }
  }
}

void ConvertConnected(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Connected");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_inner_product_param()) {
    const auto& caffe_param = caffe_layer.inner_product_param();
    if (caffe_param.has_num_output()) {
      set_s_i(shadow_op, "num_output", caffe_param.num_output());
    }
    if (caffe_param.has_bias_term()) {
      set_s_i(shadow_op, "bias_term", caffe_param.bias_term());
    }
    if (caffe_param.has_transpose()) {
      set_s_i(shadow_op, "transpose", caffe_param.bias_term());
    }
  }
}

void ConvertConv(const caffe::NetParameter& caffe_model,
                 const caffe::LayerParameter& caffe_layer,
                 shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Conv");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_convolution_param()) {
    const auto& caffe_param = caffe_layer.convolution_param();
    if (caffe_param.has_num_output()) {
      set_s_i(shadow_op, "num_output", caffe_param.num_output());
    }
    if (caffe_param.kernel_size_size() > 0) {
      set_s_i(shadow_op, "kernel_size", caffe_param.kernel_size(0));
    }
    if (caffe_param.stride_size() > 0) {
      set_s_i(shadow_op, "stride", caffe_param.stride(0));
    }
    if (caffe_param.pad_size() > 0) {
      set_s_i(shadow_op, "pad", caffe_param.pad(0));
    }
    if (caffe_param.dilation_size() > 0) {
      set_s_i(shadow_op, "dilation", caffe_param.dilation(0));
    }
    if (caffe_param.has_bias_term() && !caffe_param.bias_term()) {
      set_s_i(shadow_op, "bias_term", false);
    }
    if (caffe_param.has_group() && caffe_param.group() != 1) {
      set_s_i(shadow_op, "group", caffe_param.group());
    }
  }
}

void ConvertEltwise(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Eltwise");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  // Prod: 0, Sum: 1, Max: 2
  if (caffe_layer.has_eltwise_param()) {
    const auto& caffe_param = caffe_layer.eltwise_param();
    if (caffe_param.has_operation()) {
      if (caffe_param.operation() == caffe::EltwiseParameter_EltwiseOp_PROD) {
        set_s_i(shadow_op, "operation", 0);
      } else if (caffe_param.operation() ==
                 caffe::EltwiseParameter_EltwiseOp_SUM) {
        set_s_i(shadow_op, "operation", 1);
      } else if (caffe_param.operation() ==
                 caffe::EltwiseParameter_EltwiseOp_MAX) {
        set_s_i(shadow_op, "operation", 2);
      }
    }
    std::vector<float> coeff;
    for (const auto coe : caffe_param.coeff()) {
      coeff.push_back(coe);
    }
    set_v_f(shadow_op, "coeff", coeff);
  }
}

void ConvertFlatten(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Flatten");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_flatten_param()) {
    const auto& caffe_param = caffe_layer.flatten_param();
    if (caffe_param.has_axis()) {
      set_s_i(shadow_op, "axis", caffe_param.axis());
    }
    if (caffe_param.has_end_axis()) {
      set_s_i(shadow_op, "end_axis", caffe_param.end_axis());
    }
  }
}

void ConvertLRN(const caffe::NetParameter& caffe_model,
                const caffe::LayerParameter& caffe_layer,
                shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("LRN");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  // AcrossChannels: 0, WithinChannel: 1
  if (caffe_layer.has_lrn_param()) {
    const auto& caffe_param = caffe_layer.lrn_param();
    if (caffe_param.has_local_size()) {
      set_s_i(shadow_op, "local_size", caffe_param.local_size());
    }
    if (caffe_param.has_alpha()) {
      set_s_f(shadow_op, "alpha", caffe_param.alpha());
    }
    if (caffe_param.has_beta()) {
      set_s_f(shadow_op, "beta", caffe_param.beta());
    }
    if (caffe_param.has_norm_region()) {
      if (caffe_param.norm_region() ==
          caffe::LRNParameter_NormRegion_ACROSS_CHANNELS) {
        set_s_i(shadow_op, "norm_region", 0);
      } else {
        LOG(FATAL) << "Unsupported norm region method: Within Channel!";
      }
    }
    if (caffe_param.has_k()) {
      set_s_f(shadow_op, "k", caffe_param.k());
    }
  }
}

void ConvertNormalize(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Normalize");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_norm_param()) {
    const auto& caffe_param = caffe_layer.norm_param();
    if (caffe_param.has_across_spatial()) {
      set_s_i(shadow_op, "across_spatial", caffe_param.across_spatial());
    }
    if (caffe_param.has_channel_shared()) {
      set_s_i(shadow_op, "channel_shared", caffe_param.channel_shared());
    }
  }
}

void ConvertPermute(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Permute");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_permute_param()) {
    const auto& caffe_param = caffe_layer.permute_param();
    std::vector<int> order;
    for (const auto o : caffe_param.order()) {
      order.push_back(o);
    }
    set_v_i(shadow_op, "order", order);
  }
}

void ConvertPooling(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Pooling");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  // Max: 0, Ave: 1
  if (caffe_layer.has_pooling_param()) {
    const auto& caffe_param = caffe_layer.pooling_param();
    if (caffe_param.has_pool()) {
      if (caffe_param.pool() == caffe::PoolingParameter_PoolMethod_MAX) {
        set_s_i(shadow_op, "pool", 0);
      } else if (caffe_param.pool() == caffe::PoolingParameter_PoolMethod_AVE) {
        set_s_i(shadow_op, "pool", 1);
      }
    }
    if (caffe_param.has_kernel_size()) {
      set_s_i(shadow_op, "kernel_size", caffe_param.kernel_size());
    }
    if (caffe_param.has_stride()) {
      set_s_i(shadow_op, "stride", caffe_param.stride());
    }
    if (caffe_param.has_pad()) {
      set_s_i(shadow_op, "pad", caffe_param.pad());
    }
    if (caffe_param.has_global_pooling()) {
      set_s_i(shadow_op, "global_pooling", caffe_param.global_pooling());
    }
  }
}

void ConvertPriorBox(const caffe::NetParameter& caffe_model,
                     const caffe::LayerParameter& caffe_layer,
                     shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("PriorBox");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_prior_box_param()) {
    const auto& caffe_param = caffe_layer.prior_box_param();
    std::vector<float> min_size, max_size, aspect_ratio, variance;
    for (const auto ms : caffe_param.min_size()) {
      min_size.push_back(ms);
    }
    set_v_f(shadow_op, "min_size", min_size);
    for (const auto ms : caffe_param.max_size()) {
      max_size.push_back(ms);
    }
    set_v_f(shadow_op, "max_size", max_size);
    for (const auto ar : caffe_param.aspect_ratio()) {
      aspect_ratio.push_back(ar);
    }
    set_v_f(shadow_op, "aspect_ratio", aspect_ratio);
    if (caffe_param.has_flip()) {
      set_s_i(shadow_op, "flip", caffe_param.flip());
    }
    if (caffe_param.has_clip()) {
      set_s_i(shadow_op, "clip", caffe_param.clip());
    }
    for (const auto v : caffe_param.variance()) {
      variance.push_back(v);
    }
    set_v_f(shadow_op, "variance", variance);
    if (caffe_param.has_step()) {
      set_s_f(shadow_op, "step", caffe_param.step());
    }
    if (caffe_param.has_offset()) {
      set_s_f(shadow_op, "offset", caffe_param.offset());
    }
  }
}

void ConvertReshape(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Reshape");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_reshape_param()) {
    const auto& caffe_param = caffe_layer.reshape_param();
    std::vector<int> shape;
    for (const auto dim : caffe_param.shape().dim()) {
      shape.push_back(dim);
    }
    set_v_i(shadow_op, "shape", shape);
    if (caffe_param.has_axis()) {
      set_s_i(shadow_op, "axis", caffe_param.axis());
    }
    if (caffe_param.has_num_axes()) {
      set_s_i(shadow_op, "num_axes", caffe_param.num_axes());
    }
  }
}

void ConvertScale(const caffe::NetParameter& caffe_model,
                  const caffe::LayerParameter& caffe_layer,
                  shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Scale");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_scale_param()) {
    const auto& caffe_param = caffe_layer.scale_param();
    if (caffe_param.has_axis()) {
      set_s_i(shadow_op, "axis", caffe_param.axis());
    }
    if (caffe_param.has_num_axes()) {
      set_s_i(shadow_op, "num_axes", caffe_param.num_axes());
    }
    if (caffe_param.has_bias_term()) {
      set_s_i(shadow_op, "bias_term", caffe_param.bias_term());
    }
  }
}

void ConvertSoftmax(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Softmax");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_softmax_param()) {
    const auto& caffe_param = caffe_layer.softmax_param();
    if (caffe_param.has_axis()) {
      set_s_i(shadow_op, "axis", caffe_param.axis());
    }
  }
}

void Convert(const caffe::NetParameter& caffe_deploy,
             const caffe::NetParameter& caffe_model,
             const std::vector<int>& input_shape,
             shadow::NetParam* shadow_net) {
  shadow_net->set_name(caffe_deploy.name());
  int start_layer = 0;
  if (caffe_deploy.input_size() > 0) {
    ConvertInput(caffe_deploy, input_shape, shadow_net);
  } else {
    ConvertData(caffe_deploy.layer(0), input_shape, shadow_net);
    start_layer = 1;
  }
  for (int l = start_layer; l < caffe_deploy.layer_size(); ++l) {
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
    } else if (!layer_type.compare("Eltwise")) {
      ConvertEltwise(caffe_model, caffe_layer, shadow_net);
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

void WriteDefines(const shadow::NetParam& shadow_net, const std::string& root,
                  const std::string& model_name) {
  auto class_name = model_name;
  std::transform(model_name.begin(), model_name.end(), class_name.begin(),
                 ::toupper);
  const auto& model_name_cpp = model_name + ".cpp";
  const auto& model_name_hpp = model_name + ".hpp";
  const auto& model_name_weights_hpp = model_name + "_weights.hpp";

  int whole_count = 0;
  std::vector<int> weight_counts;
  shadow::NetParam net(shadow_net);
  for (int l = 0; l < net.op_size(); ++l) {
    auto op_param = net.mutable_op(l);
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
    ss << "weight_" << n << "_";
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

  for (int n = 0; n < weight_names.size(); ++n) {
    weight_file << "extern const float " << weight_names[n] << "[];\n";
  }
  weight_file << "\n";

  weight_file << "#endif  // SHADOW_" << class_name << "_WEIGHTS_HPP\n";

  weight_file.close();
}

void WriteWeights(const shadow::NetParam& shadow_net, const std::string& root,
                  const std::string& model_name) {
  const auto& model_name_weights_hpp = model_name + "_weights.hpp";

  int weight_count = 0;
  for (const auto& op_param : shadow_net.op()) {
    if (op_param.blobs_size() == 0) continue;

    const auto& weight_name = Util::find_replace(op_param.name(), "/", "_");
    std::ofstream file(root + "/" + model_name + "_" + weight_name + ".cpp");

    file << "#include \"" << model_name_weights_hpp << "\"\n\n";

    file << "const float weight_" << weight_count << "_[] = {\n";
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

}  // namespace Caffe2Shadow

}  // namespace Shadow

#endif  // SHADOW_TOOLS_CAFFE2SHADOW_HPP
