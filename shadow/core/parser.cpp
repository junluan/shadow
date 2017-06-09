#include "parser.hpp"
#include "util/log.hpp"

namespace Shadow {

#if !defined(USE_Protobuf)
namespace Parser {

void ParseNet(const std::string &proto_text, shadow::NetParameter *net) {
  const auto &document = Json::GetDocument(proto_text);

  const auto &net_name = Json::GetString(document, "name", "");
  const auto &json_layers = Json::GetValue(document, "layer");

  net->set_name(net_name);

  for (int i = 0; i < json_layers.Size(); ++i) {
    const auto &json_layer = json_layers[i];
    const auto &layer_name = Json::GetString(json_layer, "name", "");
    const auto &layer_type = Json::GetString(json_layer, "type", "");
    if (!layer_type.compare("Activate")) {
      net->add_layer(ParseActivate(json_layer));
    } else if (!layer_type.compare("BatchNorm")) {
      net->add_layer(ParseBatchNorm(json_layer));
    } else if (!layer_type.compare("Bias")) {
      net->add_layer(ParseBias(json_layer));
    } else if (!layer_type.compare("Concat")) {
      net->add_layer(ParseConcat(json_layer));
    } else if (!layer_type.compare("Connected")) {
      net->add_layer(ParseConnected(json_layer));
    } else if (!layer_type.compare("Convolution")) {
      net->add_layer(ParseConvolution(json_layer));
    } else if (!layer_type.compare("Data")) {
      net->add_layer(ParseData(json_layer));
    } else if (!layer_type.compare("Eltwise")) {
      net->add_layer(ParseEltwise(json_layer));
    } else if (!layer_type.compare("Flatten")) {
      net->add_layer(ParseFlatten(json_layer));
    } else if (!layer_type.compare("LRN")) {
      net->add_layer(ParseLRN(json_layer));
    } else if (!layer_type.compare("Normalize")) {
      net->add_layer(ParseNormalize(json_layer));
    } else if (!layer_type.compare("Permute")) {
      net->add_layer(ParsePermute(json_layer));
    } else if (!layer_type.compare("Pooling")) {
      net->add_layer(ParsePooling(json_layer));
    } else if (!layer_type.compare("PriorBox")) {
      net->add_layer(ParsePriorBox(json_layer));
    } else if (!layer_type.compare("Reorg")) {
      net->add_layer(ParseReorg(json_layer));
    } else if (!layer_type.compare("Reshape")) {
      net->add_layer(ParseReshape(json_layer));
    } else if (!layer_type.compare("Scale")) {
      net->add_layer(ParseScale(json_layer));
    } else if (!layer_type.compare("Softmax")) {
      net->add_layer(ParseSoftmax(json_layer));
    } else {
      LOG(FATAL) << "Error when parsing layer: " << layer_name
                 << ", layer type: " << layer_type << " is not recognized!";
    }
  }
}

void ParseCommon(const JValue &root, shadow::LayerParameter *layer) {
  layer->set_name(Json::GetString(root, "name", ""));
  layer->set_type(Json::GetString(root, "type", ""));
  for (const auto &top : Json::GetVecString(root, "top")) {
    layer->add_top(top);
  }
  for (const auto &bottom : Json::GetVecString(root, "bottom")) {
    layer->add_bottom(bottom);
  }

  if (root.HasMember("blobs")) {
    const auto &blobs = root["blobs"];
    CHECK(blobs.IsArray());
    for (int i = 0; i < blobs.Size(); ++i) {
      const auto &blob = blobs[i];
      shadow::Blob shadow_blob;
      if (blob.HasMember("shape")) {
        const auto &dims = Json::GetVecInt(blob["shape"], "dim");
        for (const auto &dim : dims) {
          shadow_blob.mutable_shape()->add_dim(dim);
        }
      }
      layer->add_blobs(shadow_blob);
    }
  }
}

const shadow::LayerParameter ParseActivate(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_activate_param();

  ParseCommon(root, &shadow_layer);

  shadow::ActivateParameter_ActivateType type =
      shadow::ActivateParameter_ActivateType_Relu;
  bool channel_shared = false;
  if (root.HasMember("activateParam")) {
    const auto &json_param = Json::GetValue(root, "activateParam");
    const auto &type_str = Json::GetString(json_param, "type", "Relu");
    if (!type_str.compare("Linear")) {
      type = shadow::ActivateParameter_ActivateType_Linear;
    } else if (!type_str.compare("Relu")) {
      type = shadow::ActivateParameter_ActivateType_Relu;
    } else if (!type_str.compare("Leaky")) {
      type = shadow::ActivateParameter_ActivateType_Leaky;
    } else if (!type_str.compare("PRelu")) {
      type = shadow::ActivateParameter_ActivateType_PRelu;
      channel_shared = Json::GetBool(json_param, "channelShared", false);
    }
  }

  shadow_param->set_type(type);
  shadow_param->set_channel_shared(channel_shared);

  return shadow_layer;
}

const shadow::LayerParameter ParseBatchNorm(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_batch_norm_param();

  ParseCommon(root, &shadow_layer);

  bool use_global_stats = true;
  if (root.HasMember("batchNormParam")) {
    const auto &json_param = Json::GetValue(root, "batchNormParam");
    use_global_stats = Json::GetBool(json_param, "useGlobalStats", true);
  }

  shadow_param->set_use_global_stats(use_global_stats);

  return shadow_layer;
}

const shadow::LayerParameter ParseBias(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_bias_param();

  ParseCommon(root, &shadow_layer);

  int axis = 1, num_axes = 1;
  if (root.HasMember("biasParam")) {
    const auto &json_param = Json::GetValue(root, "biasParam");
    axis = Json::GetInt(json_param, "axis", 1);
    num_axes = Json::GetInt(json_param, "numAxes", 1);
  }

  shadow_param->set_axis(axis);
  shadow_param->set_num_axes(num_axes);

  return shadow_layer;
}

const shadow::LayerParameter ParseConcat(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_concat_param();

  ParseCommon(root, &shadow_layer);

  int axis = 1;
  if (root.HasMember("concatParam")) {
    const auto &json_param = Json::GetValue(root, "concatParam");
    axis = Json::GetInt(json_param, "axis", 1);
  }

  shadow_param->set_axis(axis);

  return shadow_layer;
}

const shadow::LayerParameter ParseConnected(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_connected_param();

  ParseCommon(root, &shadow_layer);

  int num_output = -1;
  bool bias_term = true, transpose = false;
  if (root.HasMember("connectedParam")) {
    const auto &json_param = Json::GetValue(root, "connectedParam");
    num_output = Json::GetInt(json_param, "numOutput", -1);
    bias_term = Json::GetBool(json_param, "biasTerm", true);
    transpose = Json::GetBool(json_param, "transpose", false);
  }

  CHECK_GT(num_output, 0);
  shadow_param->set_num_output(num_output);
  shadow_param->set_bias_term(bias_term);
  shadow_param->set_transpose(transpose);

  return shadow_layer;
}

const shadow::LayerParameter ParseConvolution(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_convolution_param();

  ParseCommon(root, &shadow_layer);

  int num_output = -1, kernel_size = -1, stride = 1, pad = 0, group = 1,
      dilation = 1;
  bool bias_term = true;
  if (root.HasMember("convolutionParam")) {
    const auto &json_param = Json::GetValue(root, "convolutionParam");
    num_output = Json::GetInt(json_param, "numOutput", -1);
    kernel_size = Json::GetInt(json_param, "kernelSize", -1);
    stride = Json::GetInt(json_param, "stride", 1);
    pad = Json::GetInt(json_param, "pad", 0);
    dilation = Json::GetInt(json_param, "dilation", 1);
    group = Json::GetInt(json_param, "group", 1);
    bias_term = Json::GetBool(json_param, "biasTerm", true);
  }

  CHECK_GT(num_output, 0);
  CHECK_GT(kernel_size, 0);
  shadow_param->set_num_output(num_output);
  shadow_param->set_kernel_size(kernel_size);
  shadow_param->set_stride(stride);
  shadow_param->set_pad(pad);
  shadow_param->set_dilation(dilation);
  shadow_param->set_group(group);
  shadow_param->set_bias_term(bias_term);

  return shadow_layer;
}

const shadow::LayerParameter ParseData(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_data_param();

  ParseCommon(root, &shadow_layer);

  VecInt data_shape;
  float scale = 1;
  VecFloat mean_value;
  if (root.HasMember("dataParam")) {
    const auto &json_param = Json::GetValue(root, "dataParam");
    if (json_param.HasMember("dataShape")) {
      const auto &value = Json::GetValue(json_param, "dataShape");
      data_shape = Json::GetVecInt(value, "dim");
    }
    scale = Json::GetFloat(json_param, "scale", 1);
    mean_value = Json::GetVecFloat(json_param, "meanValue");
  }

  for (const auto &dim : data_shape) {
    shadow_param->mutable_data_shape()->add_dim(dim);
  }
  shadow_param->set_scale(scale);
  for (const auto &mean : mean_value) {
    shadow_param->add_mean_value(mean);
  }

  return shadow_layer;
}

const shadow::LayerParameter ParseEltwise(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_eltwise_param();

  ParseCommon(root, &shadow_layer);

  shadow::EltwiseParameter_EltwiseOp operation =
      shadow::EltwiseParameter_EltwiseOp_Sum;
  VecFloat coeffs;
  if (root.HasMember("eltwiseParam")) {
    const auto &json_param = Json::GetValue(root, "eltwiseParam");
    const auto &operation_str = Json::GetString(json_param, "operation", "Sum");
    if (!operation_str.compare("Prod")) {
      operation = shadow::EltwiseParameter_EltwiseOp_Prod;
    } else if (!operation_str.compare("Sum")) {
      operation = shadow::EltwiseParameter_EltwiseOp_Sum;
    } else if (!operation_str.compare("Max")) {
      operation = shadow::EltwiseParameter_EltwiseOp_Max;
    }
    coeffs = Json::GetVecFloat(json_param, "coeff");
  }

  shadow_param->set_operation(operation);
  for (const auto &coeff : coeffs) {
    shadow_param->add_coeff(coeff);
  }

  return shadow_layer;
}

const shadow::LayerParameter ParseFlatten(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_flatten_param();

  ParseCommon(root, &shadow_layer);

  int axis = 1, end_axis = -1;
  if (root.HasMember("flattenParam")) {
    const auto &json_param = Json::GetValue(root, "flattenParam");
    axis = Json::GetInt(json_param, "axis", 1);
    end_axis = Json::GetInt(json_param, "endAxis", -1);
  }

  shadow_param->set_axis(axis);
  shadow_param->set_end_axis(end_axis);

  return shadow_layer;
}

const shadow::LayerParameter ParseLRN(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_lrn_param();

  ParseCommon(root, &shadow_layer);

  shadow::LRNParameter_NormRegion norm_region =
      shadow::LRNParameter_NormRegion_AcrossChannels;
  int local_size = 5;
  float alpha = 1, beta = 0.75, k = 1;
  if (root.HasMember("lrnParam")) {
    const auto &json_param = Json::GetValue(root, "lrnParam");
    local_size = Json::GetInt(json_param, "localSize", 5);
    alpha = Json::GetFloat(json_param, "alpha", 1);
    beta = Json::GetFloat(json_param, "beta", 0.75);
    k = Json::GetFloat(json_param, "k", 1);
    const auto &norm_region_str =
        Json::GetString(json_param, "normRegion", "AcrossChannels");
    if (!norm_region_str.compare("AcrossChannels")) {
      norm_region = shadow::LRNParameter_NormRegion_AcrossChannels;
    } else if (!norm_region_str.compare("WithinChannel")) {
      norm_region = shadow::LRNParameter_NormRegion_WithinChannel;
    }
  }

  shadow_param->set_local_size(local_size);
  shadow_param->set_alpha(alpha);
  shadow_param->set_beta(beta);
  shadow_param->set_norm_region(norm_region);
  shadow_param->set_k(k);

  return shadow_layer;
}

const shadow::LayerParameter ParseNormalize(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_normalize_param();

  ParseCommon(root, &shadow_layer);

  bool across_spatial = true, channel_shared = true;
  VecFloat scale;
  if (root.HasMember("normalizeParam")) {
    const auto &json_param = Json::GetValue(root, "normalizeParam");
    across_spatial = Json::GetBool(json_param, "acrossSpatial", true);
    channel_shared = Json::GetBool(json_param, "channelShared", true);
    scale = Json::GetVecFloat(json_param, "scale");
  }

  shadow_param->set_across_spatial(across_spatial);
  shadow_param->set_channel_shared(channel_shared);
  for (const auto &s : scale) {
    shadow_param->add_scale(s);
  }

  return shadow_layer;
}

const shadow::LayerParameter ParsePermute(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_permute_param();

  ParseCommon(root, &shadow_layer);

  VecInt order;
  if (root.HasMember("permuteParam")) {
    const auto &json_param = Json::GetValue(root, "permuteParam");
    order = Json::GetVecInt(json_param, "order");
  }

  for (const auto &o : order) {
    shadow_param->add_order(o);
  }

  return shadow_layer;
}

const shadow::LayerParameter ParsePooling(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_pooling_param();

  ParseCommon(root, &shadow_layer);

  shadow::PoolingParameter_PoolType pool =
      shadow::PoolingParameter_PoolType_Max;
  int kernel_size = -1, stride = 1, pad = 0;
  bool global_pooling = false;
  if (root.HasMember("poolingParam")) {
    const auto &json_param = Json::GetValue(root, "poolingParam");
    kernel_size = Json::GetInt(json_param, "kernelSize", -1);
    stride = Json::GetInt(json_param, "stride", 1);
    pad = Json::GetInt(json_param, "pad", 0);
    global_pooling = Json::GetBool(json_param, "globalPooling", false);
    const auto &pool_str = Json::GetString(json_param, "pool", "Max");
    if (!pool_str.compare("Max")) {
      pool = shadow::PoolingParameter_PoolType_Max;
    } else if (!pool_str.compare("Ave")) {
      pool = shadow::PoolingParameter_PoolType_Ave;
    }
  }

  if (!global_pooling) {
    CHECK_GT(kernel_size, 0);
  }
  shadow_param->set_pool(pool);
  shadow_param->set_kernel_size(kernel_size);
  shadow_param->set_stride(stride);
  shadow_param->set_pad(pad);
  shadow_param->set_global_pooling(global_pooling);

  return shadow_layer;
}

const shadow::LayerParameter ParsePriorBox(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_prior_box_param();

  ParseCommon(root, &shadow_layer);

  VecFloat min_size, max_size, aspect_ratio, variance;
  bool flip = true, clip = false;
  float step = -1, offset = 0.5;
  if (root.HasMember("priorBoxParam")) {
    const auto &json_param = Json::GetValue(root, "priorBoxParam");
    min_size = Json::GetVecFloat(json_param, "minSize");
    max_size = Json::GetVecFloat(json_param, "maxSize");
    aspect_ratio = Json::GetVecFloat(json_param, "aspectRatio");
    flip = Json::GetBool(json_param, "flip", true);
    clip = Json::GetBool(json_param, "clip", false);
    variance = Json::GetVecFloat(json_param, "variance");
    step = Json::GetFloat(json_param, "step", -1);
    offset = Json::GetFloat(json_param, "offset", 0.5);
  }

  for (const auto &min : min_size) {
    shadow_param->add_min_size(min);
  }
  for (const auto &max : max_size) {
    shadow_param->add_max_size(max);
  }
  for (const auto &r : aspect_ratio) {
    shadow_param->add_aspect_ratio(r);
  }
  shadow_param->set_flip(flip);
  shadow_param->set_clip(clip);
  for (const auto &v : variance) {
    shadow_param->add_variance(v);
  }
  if (step > 0) {
    shadow_param->set_step(step);
  }
  shadow_param->set_offset(offset);

  return shadow_layer;
}

const shadow::LayerParameter ParseReorg(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_reorg_param();

  ParseCommon(root, &shadow_layer);

  int stride = 2;
  if (root.HasMember("reorgParam")) {
    const auto &json_param = Json::GetValue(root, "reorgParam");
    stride = Json::GetInt(json_param, "stride", 2);
  }

  shadow_param->set_stride(stride);

  return shadow_layer;
}

const shadow::LayerParameter ParseReshape(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_reshape_param();

  ParseCommon(root, &shadow_layer);

  VecInt shape;
  int axis = 0, num_axes = -1;
  if (root.HasMember("reshapeParam")) {
    const auto &json_param = Json::GetValue(root, "reshapeParam");
    if (json_param.HasMember("shape")) {
      const auto &value = Json::GetValue(json_param, "shape");
      shape = Json::GetVecInt(value, "dim");
    }
    axis = Json::GetInt(json_param, "axis", 0);
    num_axes = Json::GetInt(json_param, "numAxes", -1);
  }

  for (const auto &dim : shape) {
    shadow_param->mutable_shape()->add_dim(dim);
  }
  shadow_param->set_axis(axis);
  shadow_param->set_num_axes(num_axes);

  return shadow_layer;
}

const shadow::LayerParameter ParseScale(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_scale_param();

  ParseCommon(root, &shadow_layer);

  int axis = 1, num_axes = 1;
  bool bias_term = false;
  if (root.HasMember("scaleParam")) {
    const auto &json_param = Json::GetValue(root, "scaleParam");
    axis = Json::GetInt(json_param, "axis", 1);
    num_axes = Json::GetInt(json_param, "numAxes", 1);
    bias_term = Json::GetBool(json_param, "biasTerm", false);
  }

  shadow_param->set_axis(axis);
  shadow_param->set_num_axes(num_axes);
  shadow_param->set_bias_term(bias_term);

  return shadow_layer;
}

const shadow::LayerParameter ParseSoftmax(const JValue &root) {
  shadow::LayerParameter shadow_layer;
  auto shadow_param = shadow_layer.mutable_softmax_param();

  ParseCommon(root, &shadow_layer);

  int axis = 1;
  if (root.HasMember("softmaxParam")) {
    const auto &json_param = Json::GetValue(root, "softmaxParam");
    axis = Json::GetInt(json_param, "axis", 1);
  }

  shadow_param->set_axis(axis);

  return shadow_layer;
}

}  // namespace Parser
#endif

}  // namespace Shadow
