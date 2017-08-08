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
  if (caffe_layer.type() == "ReLU") {
    set_s_i(shadow_op, "type", 1);
  } else if (caffe_layer.type() == "PReLU") {
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

void ConvertCaffe(const caffe::NetParameter& caffe_deploy,
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
    if (layer_type == "ReLU" || layer_type == "PReLU") {
      ConvertActivate(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "BatchNorm") {
      ConvertBatchNorm(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Bias") {
      ConvertBias(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Concat") {
      ConvertConcat(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "InnerProduct") {
      ConvertConnected(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Convolution") {
      ConvertConv(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Eltwise") {
      ConvertEltwise(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Flatten") {
      ConvertFlatten(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "LRN") {
      ConvertLRN(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Normalize") {
      ConvertNormalize(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Permute") {
      ConvertPermute(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Pooling") {
      ConvertPooling(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "PriorBox") {
      ConvertPriorBox(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Reshape") {
      ConvertReshape(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Scale") {
      ConvertScale(caffe_model, caffe_layer, shadow_net);
    } else if (layer_type == "Softmax") {
      ConvertSoftmax(caffe_model, caffe_layer, shadow_net);
    } else {
      LOG(WARNING) << "Layer type: " << layer_type << " is not recognized!";
    }
  }
}

}  // namespace Caffe2Shadow

}  // namespace Shadow

#endif  // SHADOW_TOOLS_CAFFE2SHADOW_HPP
