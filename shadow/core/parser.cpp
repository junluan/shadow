#include "parser.hpp"
#include "util/log.hpp"

#include <functional>

namespace Shadow {

namespace Parser {

#if !defined(USE_Protobuf)
using ParseFunc = std::function<const shadow::OpParam(const JValue &)>;

static const std::map<std::string, ParseFunc> parse_func_map{
    {"Activate", ParseActivate},   {"BatchNorm", ParseBatchNorm},
    {"Bias", ParseBias},           {"Concat", ParseConcat},
    {"Connected", ParseConnected}, {"Conv", ParseConv},
    {"Data", ParseData},           {"Eltwise", ParseEltwise},
    {"Flatten", ParseFlatten},     {"LRN", ParseLRN},
    {"Normalize", ParseNormalize}, {"Permute", ParsePermute},
    {"Pooling", ParsePooling},     {"PriorBox", ParsePriorBox},
    {"Reorg", ParseReorg},         {"Reshape", ParseReshape},
    {"Scale", ParseScale},         {"Softmax", ParseSoftmax}};

void ParseNet(const std::string &proto_text, shadow::NetParam *net) {
  const auto &document = Json::GetDocument(proto_text);

  const auto &net_name = Json::GetString(document, "name", "");
  const auto &json_ops = Json::GetValue(document, "op");

  net->set_name(net_name);

  for (int i = 0; i < json_ops.Size(); ++i) {
    const auto &json_op = json_ops[i];
    const auto &op_name = Json::GetString(json_op, "name", "");
    const auto &op_type = Json::GetString(json_op, "type", "");

    bool find_parser = false;
    for (const auto &it : parse_func_map) {
      if (op_type.find(it.first) != std::string::npos) {
        net->add_op(it.second(json_op));
        find_parser = true;
        break;
      }
    }
    CHECK(find_parser) << "Error when parsing op: " << op_name
                       << ", op type: " << op_type << " is not recognized!";
  }
}

void ParseCommon(const JValue &root, shadow::OpParam *op) {
  op->set_name(Json::GetString(root, "name", ""));
  op->set_type(Json::GetString(root, "type", ""));
  for (const auto &top : Json::GetVecString(root, "top")) {
    op->add_top(top);
  }
  for (const auto &bottom : Json::GetVecString(root, "bottom")) {
    op->add_bottom(bottom);
  }

  if (root.HasMember("blobs")) {
    const auto &blobs = root["blobs"];
    CHECK(blobs.IsArray());
    for (int i = 0; i < blobs.Size(); ++i) {
      const auto &blob = blobs[i];
      shadow::Blob shadow_blob;
      if (blob.HasMember("shape")) {
        const auto &shape = Json::GetVecInt(blob, "shape");
        for (const auto &dim : shape) {
          shadow_blob.add_shape(dim);
        }
      }
      op->add_blobs(shadow_blob);
    }
  }
}

const shadow::OpParam ParseActivate(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int type = 1, channel_shared = false;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "type") {
        type = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "channel_shared") {
        channel_shared = Json::GetInt(arg, "s_i", 0);
      }
    }
  }

  set_s_i(&shadow_op, "type", type);
  set_s_i(&shadow_op, "channel_shared", channel_shared);

  return shadow_op;
}

const shadow::OpParam ParseBatchNorm(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int use_global_stats = true;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "use_global_stats") {
        use_global_stats = Json::GetInt(arg, "s_i", 1);
      }
    }
  }

  set_s_i(&shadow_op, "use_global_stats", use_global_stats);

  return shadow_op;
}

const shadow::OpParam ParseBias(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int axis = 1, num_axes = 1;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "num_axes") {
        num_axes = Json::GetInt(arg, "s_i", 1);
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);
  set_s_i(&shadow_op, "num_axes", num_axes);

  return shadow_op;
}

const shadow::OpParam ParseConcat(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int axis = 1;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 1);
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);

  return shadow_op;
}

const shadow::OpParam ParseConnected(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int num_output = -1, bias_term = true, transpose = false;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "num_output") {
        num_output = Json::GetInt(arg, "s_i", -1);
      } else if (arg_name == "bias_term") {
        bias_term = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "transpose") {
        transpose = Json::GetInt(arg, "s_i", 0);
      }
    }
  }

  CHECK_GT(num_output, 0);
  set_s_i(&shadow_op, "num_output", num_output);
  set_s_i(&shadow_op, "bias_term", bias_term);
  set_s_i(&shadow_op, "transpose", transpose);

  return shadow_op;
}

const shadow::OpParam ParseConv(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int num_output = -1, kernel_size = -1, stride = 1, pad = 0, dilation = 1,
      group = 1, bias_term = true;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "num_output") {
        num_output = Json::GetInt(arg, "s_i", -1);
      } else if (arg_name == "kernel_size") {
        kernel_size = Json::GetInt(arg, "s_i", -1);
      } else if (arg_name == "stride") {
        stride = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "pad") {
        pad = Json::GetInt(arg, "s_i", 0);
      } else if (arg_name == "dilation") {
        dilation = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "group") {
        group = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "bias_term") {
        bias_term = Json::GetInt(arg, "s_i", 1);
      }
    }
  }

  CHECK_GT(num_output, 0);
  CHECK_GT(kernel_size, 0);
  set_s_i(&shadow_op, "num_output", num_output);
  set_s_i(&shadow_op, "kernel_size", kernel_size);
  set_s_i(&shadow_op, "stride", stride);
  set_s_i(&shadow_op, "pad", pad);
  set_s_i(&shadow_op, "dilation", dilation);
  set_s_i(&shadow_op, "group", group);
  set_s_i(&shadow_op, "bias_term", bias_term);

  return shadow_op;
}

const shadow::OpParam ParseData(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  VecInt data_shape;
  float scale = 1;
  VecFloat mean_value;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "data_shape") {
        data_shape = Json::GetVecInt(arg, "v_i");
      } else if (arg_name == "scale") {
        scale = Json::GetFloat(arg, "s_f", 1);
      } else if (arg_name == "mean_value") {
        mean_value = Json::GetVecFloat(arg, "v_f");
      }
    }
  }

  set_v_i(&shadow_op, "data_shape", data_shape);
  set_s_f(&shadow_op, "scale", scale);
  set_v_f(&shadow_op, "mean_value", mean_value);

  return shadow_op;
}

const shadow::OpParam ParseEltwise(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int operation = 1;
  VecFloat coeffs;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "operation") {
        operation = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "coeff") {
        coeffs = Json::GetVecFloat(arg, "v_f");
      }
    }
  }

  set_s_i(&shadow_op, "operation", operation);
  set_v_f(&shadow_op, "coeff", coeffs);

  return shadow_op;
}

const shadow::OpParam ParseFlatten(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int axis = 1, end_axis = -1;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "end_axis") {
        end_axis = Json::GetInt(arg, "s_i", -1);
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);
  set_s_i(&shadow_op, "end_axis", end_axis);

  return shadow_op;
}

const shadow::OpParam ParseLRN(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int norm_region = 0, local_size = 5;
  float alpha = 1, beta = 0.75, k = 1;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "local_size") {
        local_size = Json::GetInt(arg, "s_i", 5);
      } else if (arg_name == "alpha") {
        alpha = Json::GetFloat(arg, "s_f", -1.f);
      } else if (arg_name == "beta") {
        beta = Json::GetFloat(arg, "s_f", 0.75f);
      } else if (arg_name == "k") {
        k = Json::GetFloat(arg, "s_f", 1);
      } else if (arg_name == "norm_region") {
        norm_region = Json::GetInt(arg, "s_i", 0);
      }
    }
  }

  set_s_i(&shadow_op, "local_size", local_size);
  set_s_f(&shadow_op, "alpha", alpha);
  set_s_f(&shadow_op, "beta", beta);
  set_s_f(&shadow_op, "k", k);
  set_s_i(&shadow_op, "norm_region", norm_region);

  return shadow_op;
}

const shadow::OpParam ParseNormalize(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int across_spatial = true, channel_shared = true;
  VecFloat scale;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "across_spatial") {
        across_spatial = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "channel_shared") {
        channel_shared = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "scale") {
        scale = Json::GetVecFloat(arg, "v_f");
      }
    }
  }

  set_s_i(&shadow_op, "across_spatial", across_spatial);
  set_s_i(&shadow_op, "channel_shared", channel_shared);
  set_v_f(&shadow_op, "scale", scale);

  return shadow_op;
}

const shadow::OpParam ParsePermute(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  VecInt order;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "order") {
        order = Json::GetVecInt(arg, "v_i");
      }
    }
  }

  set_v_i(&shadow_op, "order", order);

  return shadow_op;
}

const shadow::OpParam ParsePooling(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int pool = 0, kernel_size = -1, stride = 1, pad = 0, global_pooling = false;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "pool") {
        pool = Json::GetInt(arg, "s_i", 0);
      } else if (arg_name == "kernel_size") {
        kernel_size = Json::GetInt(arg, "s_i", -1);
      } else if (arg_name == "stride") {
        stride = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "pad") {
        pad = Json::GetInt(arg, "s_i", 0);
      } else if (arg_name == "global_pooling") {
        global_pooling = Json::GetInt(arg, "s_i", 0);
      }
    }
  }

  if (!global_pooling) {
    CHECK_GT(kernel_size, 0);
  }
  set_s_i(&shadow_op, "pool", pool);
  set_s_i(&shadow_op, "kernel_size", kernel_size);
  set_s_i(&shadow_op, "stride", stride);
  set_s_i(&shadow_op, "pad", pad);
  set_s_i(&shadow_op, "global_pooling", global_pooling);

  return shadow_op;
}

const shadow::OpParam ParsePriorBox(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  VecFloat min_size, max_size, aspect_ratio, variance;
  int flip = true, clip = false;
  float step = -1, offset = 0.5;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "min_size") {
        min_size = Json::GetVecFloat(arg, "v_f");
      } else if (arg_name == "max_size") {
        max_size = Json::GetVecFloat(arg, "v_f");
      } else if (arg_name == "aspect_ratio") {
        aspect_ratio = Json::GetVecFloat(arg, "v_f");
      } else if (arg_name == "flip") {
        flip = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "clip") {
        clip = Json::GetInt(arg, "s_i", 0);
      } else if (arg_name == "variance") {
        variance = Json::GetVecFloat(arg, "v_f");
      } else if (arg_name == "step") {
        step = Json::GetFloat(arg, "s_f", -1);
      } else if (arg_name == "offset") {
        offset = Json::GetFloat(arg, "s_f", 0.5f);
      }
    }
  }

  set_v_f(&shadow_op, "min_size", min_size);
  set_v_f(&shadow_op, "max_size", max_size);
  set_v_f(&shadow_op, "aspect_ratio", aspect_ratio);
  set_s_i(&shadow_op, "flip", flip);
  set_s_i(&shadow_op, "clip", clip);
  set_v_f(&shadow_op, "variance", variance);
  if (step > 0) {
    set_s_f(&shadow_op, "step", step);
  }
  set_s_f(&shadow_op, "offset", offset);

  return shadow_op;
}

const shadow::OpParam ParseReorg(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int stride = 2;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "stride") {
        stride = Json::GetInt(arg, "s_i", 2);
      }
    }
  }

  set_s_i(&shadow_op, "stride", stride);

  return shadow_op;
}

const shadow::OpParam ParseReshape(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  VecInt shape;
  int axis = 0, num_axes = -1;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "shape") {
        shape = Json::GetVecInt(arg, "v_i");
      } else if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 0);
      } else if (arg_name == "num_axes") {
        num_axes = Json::GetInt(arg, "s_i", -1);
      }
    }
  }

  set_v_i(&shadow_op, "shape", shape);
  set_s_i(&shadow_op, "axis", axis);
  set_s_i(&shadow_op, "num_axes", num_axes);

  return shadow_op;
}

const shadow::OpParam ParseScale(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int axis = 1, num_axes = 1, bias_term = false;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "num_axes") {
        num_axes = Json::GetInt(arg, "s_i", 0);
      } else if (arg_name == "bias_term") {
        bias_term = Json::GetInt(arg, "s_i", 0);
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);
  set_s_i(&shadow_op, "num_axes", num_axes);
  set_s_i(&shadow_op, "bias_term", bias_term);

  return shadow_op;
}

const shadow::OpParam ParseSoftmax(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int axis = 1;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 1);
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);

  return shadow_op;
}
#endif

}  // namespace Parser

}  // namespace Shadow
