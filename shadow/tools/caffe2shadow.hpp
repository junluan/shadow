#ifndef SHADOW_TOOLS_CAFFE2SHADOW_HPP
#define SHADOW_TOOLS_CAFFE2SHADOW_HPP

#include "proto/caffe.pb.h"

#include "transformer.hpp"

namespace Shadow {

namespace Caffe2Shadow {

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
    if (layer_name == layer.name()) {
      for (const auto& caffe_blob : layer.blobs()) {
        auto shadow_blob = shadow_op->add_blobs();
        if (caffe_blob.shape().dim_size() > 0) {
          for (const auto dim : caffe_blob.shape().dim()) {
            shadow_blob->add_shape(static_cast<int>(dim));
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

int ConvertInput(const caffe::NetParameter& caffe_deploy,
                 const NetInfo& net_info, shadow::NetParam* shadow_net) {
  int start_layer = 0;

  std::vector<std::string> shadow_inputs;
  std::vector<std::vector<int>> shadow_shapes;
  if (caffe_deploy.input_size() > 0) {
    for (const auto& input_name : caffe_deploy.input()) {
      shadow_inputs.push_back(input_name);
    }
    if (net_info.input_shape.empty()) {
      if (caffe_deploy.input_shape_size() > 0) {
        for (const auto& caffe_shape : caffe_deploy.input_shape()) {
          std::vector<int> shadow_shape;
          for (const auto dim : caffe_shape.dim()) {
            shadow_shape.push_back(dim);
          }
          shadow_shapes.push_back(shadow_shape);
        }
      } else if (caffe_deploy.input_dim_size() > 0) {
        CHECK_EQ(caffe_deploy.input_dim_size() % 4, 0);
        for (int n = 0; n < caffe_deploy.input_dim_size() / 4; ++n) {
          std::vector<int> shadow_shape;
          for (int i = 4 * n; i < 4 * (n + 1); ++i) {
            shadow_shape.push_back(caffe_deploy.input_dim(i));
          }
          shadow_shapes.push_back(shadow_shape);
        }
      }
    } else {
      shadow_shapes = net_info.input_shape;
    }
    start_layer = 0;
  } else if (caffe_deploy.layer(0).type() == "Input") {
    const auto& caffe_input_layer = caffe_deploy.layer(0);
    for (const auto& input_name : caffe_input_layer.top()) {
      shadow_inputs.push_back(input_name);
    }
    if (net_info.input_shape.empty()) {
      if (caffe_input_layer.has_input_param()) {
        const auto& caffe_param = caffe_input_layer.input_param();
        for (const auto& caffe_shape : caffe_param.shape()) {
          std::vector<int> shadow_shape;
          for (const auto dim : caffe_shape.dim()) {
            shadow_shape.push_back(dim);
          }
          shadow_shapes.push_back(shadow_shape);
        }
      }
    } else {
      shadow_shapes = net_info.input_shape;
    }
    start_layer = 1;
  }

  CHECK_GT(shadow_inputs.size(), 0) << "Must supply input";

  std::string data_blob_name;

  auto shadow_input_op = shadow_net->add_op();
  shadow_input_op->set_name("input");
  shadow_input_op->set_type("Input");
  for (int n = 0; n < shadow_inputs.size(); ++n) {
    const auto& input_name = shadow_inputs[n];
    if (input_name.find("data") != std::string::npos) {
      data_blob_name = input_name;
    }
    shadow_input_op->add_top(input_name);
    if (shadow_shapes.size() == shadow_inputs.size()) {
      set_v_i(shadow_input_op, input_name, shadow_shapes[n]);
    } else if (shadow_shapes.size() == 1) {
      set_v_i(shadow_input_op, input_name, shadow_shapes[0]);
    } else {
      LOG(WARNING) << "No input shape, must be supplied manually";
    }
  }

  if (net_info.scale != 1 || !net_info.mean_value.empty()) {
    if (data_blob_name.empty()) {
      LOG(FATAL) << "Data blob does not has \"data\" keyword";
    }
    auto shadow_data_op = shadow_net->add_op();
    shadow_data_op->set_name("data_transform");
    shadow_data_op->set_type("Data");
    shadow_data_op->add_bottom(data_blob_name);
    shadow_data_op->add_top(data_blob_name);
    if (net_info.scale != 1) {
      set_s_f(shadow_data_op, "scale", net_info.scale);
    }
    if (!net_info.mean_value.empty()) {
      set_v_f(shadow_data_op, "mean_value", net_info.mean_value);
    }
  }

  return start_layer;
}

void ConvertActivate(const caffe::NetParameter& caffe_model,
                     const caffe::LayerParameter& caffe_layer,
                     shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Activate");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  // PRelu: 0, Relu: 1, Leaky: 2, Sigmoid: 3, SoftPlus: 4, Tanh: 5
  if (caffe_layer.type() == "ReLU") {
    set_s_i(shadow_op, "type", 1);
  } else if (caffe_layer.type() == "PReLU") {
    set_s_i(shadow_op, "type", 0);
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
    if (caffe_param.has_eps()) {
      set_s_f(shadow_op, "eps", caffe_param.eps());
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

void ConvertDeconv(const caffe::NetParameter& caffe_model,
                   const caffe::LayerParameter& caffe_layer,
                   shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("Deconv");
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
    if (!coeff.empty()) {
      set_v_f(shadow_op, "coeff", coeff);
    }
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
    if (!order.empty()) {
      set_v_i(shadow_op, "order", order);
    }
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
    if (!min_size.empty()) {
      set_v_f(shadow_op, "min_size", min_size);
    }
    for (const auto ms : caffe_param.max_size()) {
      max_size.push_back(ms);
    }
    if (!max_size.empty()) {
      set_v_f(shadow_op, "max_size", max_size);
    }
    for (const auto ar : caffe_param.aspect_ratio()) {
      aspect_ratio.push_back(ar);
    }
    if (!aspect_ratio.empty()) {
      set_v_f(shadow_op, "aspect_ratio", aspect_ratio);
    }
    if (caffe_param.has_flip()) {
      set_s_i(shadow_op, "flip", caffe_param.flip());
    }
    if (caffe_param.has_clip()) {
      set_s_i(shadow_op, "clip", caffe_param.clip());
    }
    for (const auto v : caffe_param.variance()) {
      variance.push_back(v);
    }
    if (!variance.empty()) {
      set_v_f(shadow_op, "variance", variance);
    }
    if (caffe_param.has_step()) {
      set_s_f(shadow_op, "step", caffe_param.step());
    }
    if (caffe_param.has_offset()) {
      set_s_f(shadow_op, "offset", caffe_param.offset());
    }
  }
}

void ConvertPython(const caffe::NetParameter& caffe_model,
                   const caffe::LayerParameter& caffe_layer,
                   shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_python_param()) {
    const auto& caffe_param = caffe_layer.python_param();
    if (caffe_param.layer() == "ProposalLayer") {
      shadow_op->set_type("Proposal");
      LOG(WARNING) << "Can not parse python param, please check "
                   << caffe_param.param_str();
    } else {
      LOG(FATAL) << "Layer not support " << caffe_param.layer();
    }
  } else {
    LOG(FATAL) << "Must have python param";
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
    if (!shape.empty()) {
      set_v_i(shadow_op, "shape", shape);
    }
    if (caffe_param.has_axis()) {
      set_s_i(shadow_op, "axis", caffe_param.axis());
    }
    if (caffe_param.has_num_axes()) {
      set_s_i(shadow_op, "num_axes", caffe_param.num_axes());
    }
  }
}

void ConvertROIPooling(const caffe::NetParameter& caffe_model,
                       const caffe::LayerParameter& caffe_layer,
                       shadow::NetParam* shadow_net) {
  auto shadow_op = shadow_net->add_op();
  shadow_op->set_type("ROIPooling");
  ConvertCommon(caffe_model, caffe_layer, shadow_op);

  if (caffe_layer.has_roi_pooling_param()) {
    const auto& caffe_param = caffe_layer.roi_pooling_param();
    if (caffe_param.has_pooled_h()) {
      set_s_i(shadow_op, "pooled_h", caffe_param.pooled_h());
    }
    if (caffe_param.has_pooled_w()) {
      set_s_i(shadow_op, "pooled_w", caffe_param.pooled_w());
    }
    if (caffe_param.has_spatial_scale()) {
      set_s_f(shadow_op, "spatial_scale", caffe_param.spatial_scale());
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

using ConvertFunc =
    std::function<void(const caffe::NetParameter&, const caffe::LayerParameter&,
                       shadow::NetParam*)>;

static const std::map<std::string, ConvertFunc> convert_func_map{
    {"ReLU", ConvertActivate},
    {"PReLU", ConvertActivate},
    {"BatchNorm", ConvertBatchNorm},
    {"Bias", ConvertBias},
    {"Concat", ConvertConcat},
    {"InnerProduct", ConvertConnected},
    {"Convolution", ConvertConv},
    {"Deconvolution", ConvertDeconv},
    {"Eltwise", ConvertEltwise},
    {"Flatten", ConvertFlatten},
    {"LRN", ConvertLRN},
    {"Normalize", ConvertNormalize},
    {"Permute", ConvertPermute},
    {"Pooling", ConvertPooling},
    {"PriorBox", ConvertPriorBox},
    {"Python", ConvertPython},
    {"Reshape", ConvertReshape},
    {"ROIPooling", ConvertROIPooling},
    {"Scale", ConvertScale},
    {"Softmax", ConvertSoftmax}};

void ConvertCaffe(const std::vector<caffe::NetParameter>& caffe_deploys,
                  const std::vector<caffe::NetParameter>& caffe_models,
                  const MetaNetInfo& meta_net_info,
                  std::vector<shadow::NetParam>* shadow_nets) {
  for (int n = 0; n < caffe_deploys.size(); ++n) {
    const auto& caffe_deploy = caffe_deploys[n];
    const auto& caffe_model = caffe_models[n];
    const auto& net_info = meta_net_info.network[n];

    shadow::NetParam shadow_net;

    shadow_net.set_name(caffe_deploy.name());
    for (const auto& dim : net_info.num_class) {
      shadow_net.add_num_class(dim);
    }
    for (const auto& blob_name : net_info.out_blob) {
      shadow_net.add_out_blob(blob_name);
    }

    int start_layer = ConvertInput(caffe_deploy, net_info, &shadow_net);

    for (int l = start_layer; l < caffe_deploy.layer_size(); ++l) {
      const auto& caffe_layer = caffe_deploy.layer(l);
      const auto& layer_type = caffe_layer.type();
      if (convert_func_map.count(layer_type)) {
        convert_func_map.at(layer_type)(caffe_model, caffe_layer, &shadow_net);
      } else {
        LOG(WARNING) << "Layer type: " << layer_type << " is not recognized!";
      }
    }

    shadow_nets->push_back(shadow_net);
  }
}

}  // namespace Caffe2Shadow

}  // namespace Shadow

#endif  // SHADOW_TOOLS_CAFFE2SHADOW_HPP
