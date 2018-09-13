#include "parser.hpp"

#include "util/json.hpp"
#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

namespace Parser {

#if !defined(USE_Protobuf)

#if defined(USE_JSON)
void ParseCommon(const JValue &root, shadow::OpParam *op) {
  op->set_name(Json::GetString(root, "name", ""));
  op->set_type(Json::GetString(root, "type", ""));
  for (const auto &bottom : Json::GetVecString(root, "bottom")) {
    op->add_bottom(bottom);
  }
  for (const auto &top : Json::GetVecString(root, "top")) {
    op->add_top(top);
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

const shadow::OpParam ParseBinary(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int operation = -1;
  float scalar = 0;
  bool has_scalar = false;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "operation") {
        operation = Json::GetInt(arg, "s_i", -1);
      } else if (arg_name == "scalar") {
        scalar = Json::GetFloat(arg, "s_f", 0);
        has_scalar = true;
      }
    }
  }

  set_s_i(&shadow_op, "operation", operation);
  if (has_scalar) {
    set_s_f(&shadow_op, "scalar", scalar);
  }

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

  int num_output = -1, bias_term = true, transpose = true;
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
        transpose = Json::GetInt(arg, "s_i", 1);
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
      group = 1, bias_term = true, type = -1;
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
      } else if (arg_name == "type") {
        type = Json::GetInt(arg, "s_i", -1);
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
  set_s_i(&shadow_op, "type", type);

  return shadow_op;
}

const shadow::OpParam ParseEltwise(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int operation = 1;
  std::vector<float> coeffs;
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

const shadow::OpParam ParseInput(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      const auto &shape = Json::GetVecInt(arg, "v_i");
      set_v_i(&shadow_op, arg_name, shape);
    }
  }

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
  std::vector<float> scale;
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

  std::vector<int> order;
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

  int pool = 0, global_pooling = false;
  std::vector<int> kernel_size, stride, pad;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "pool") {
        pool = Json::GetInt(arg, "s_i", 0);
      } else if (arg_name == "kernel_size") {
        kernel_size = Json::GetVecInt(arg, "v_i");
      } else if (arg_name == "stride") {
        stride = Json::GetVecInt(arg, "v_i");
      } else if (arg_name == "pad") {
        pad = Json::GetVecInt(arg, "v_i");
      } else if (arg_name == "global_pooling") {
        global_pooling = Json::GetInt(arg, "s_i", 0);
      }
    }
  }

  set_s_i(&shadow_op, "pool", pool);
  set_v_i(&shadow_op, "kernel_size", kernel_size);
  set_v_i(&shadow_op, "stride", stride);
  set_v_i(&shadow_op, "pad", pad);
  set_s_i(&shadow_op, "global_pooling", global_pooling);

  return shadow_op;
}

const shadow::OpParam ParsePriorBox(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  std::vector<float> min_size, max_size, aspect_ratio, variance;
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

  std::vector<int> shape;
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

  std::vector<float> scale_value, bias_value;
  int axis = 1, has_scale = true, has_bias = true;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "has_scale") {
        has_scale = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "has_bias") {
        has_bias = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "scale_value") {
        scale_value = Json::GetVecFloat(arg, "v_f");
      } else if (arg_name == "bias_value") {
        bias_value = Json::GetVecFloat(arg, "v_f");
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);
  set_s_i(&shadow_op, "has_scale", has_scale);
  set_s_i(&shadow_op, "has_bias", has_bias);
  set_v_f(&shadow_op, "scale_value", scale_value);
  set_v_f(&shadow_op, "bias_value", bias_value);

  return shadow_op;
}

const shadow::OpParam ParseSlice(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int axis = 1;
  std::vector<int> slice_point;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 1);
      } else if (arg_name == "slice_point") {
        slice_point = Json::GetVecInt(arg, "v_i");
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);
  set_v_i(&shadow_op, "slice_point", slice_point);

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

const shadow::OpParam ParseStack(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int axis = 0;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "axis") {
        axis = Json::GetInt(arg, "s_i", 0);
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);

  return shadow_op;
}

const shadow::OpParam ParseUnary(const JValue &root) {
  shadow::OpParam shadow_op;

  ParseCommon(root, &shadow_op);

  int operation = -1;
  if (root.HasMember("arg")) {
    const auto &args = root["arg"];
    for (int i = 0; i < args.Size(); ++i) {
      const auto &arg = args[i];
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      if (arg_name == "operation") {
        operation = Json::GetInt(arg, "s_i", -1);
      }
    }
  }

  set_s_i(&shadow_op, "operation", operation);

  return shadow_op;
}

using ParseFunc = std::function<const shadow::OpParam(const JValue &)>;

static const std::map<std::string, ParseFunc> parse_func_map{
    {"Activate", ParseActivate},
    {"BatchNorm", ParseBatchNorm},
    {"Binary", ParseBinary},
    {"Concat", ParseConcat},
    {"Connected", ParseConnected},
    {"Conv", ParseConv},
    {"Deconv", ParseConv},
    {"Eltwise", ParseEltwise},
    {"Flatten", ParseFlatten},
    {"Input", ParseInput},
    {"LRN", ParseLRN},
    {"Normalize", ParseNormalize},
    {"Permute", ParsePermute},
    {"Pooling", ParsePooling},
    {"PriorBox", ParsePriorBox},
    {"Reorg", ParseReorg},
    {"Reshape", ParseReshape},
    {"Scale", ParseScale},
    {"Slice", ParseSlice},
    {"Softmax", ParseSoftmax},
    {"Stack", ParseStack},
    {"Unary", ParseUnary}};

void ParseNet(const std::string &proto_text, shadow::NetParam *net) {
  const auto &document = Json::GetDocument(proto_text);

  const auto &net_name = Json::GetString(document, "name", "");
  const auto &json_ops = Json::GetValue(document, "op");

  net->set_name(net_name);

  if (document.HasMember("arg")) {
    const auto &json_arg = document["arg"];
    CHECK(json_arg.IsArray());
    for (int n = 0; n < json_arg.Size(); ++n) {
      const auto &arg = json_arg[n];
      shadow::Argument shadow_arg;
      CHECK(arg.HasMember("name"));
      const auto &arg_name = Json::GetString(arg, "name", "");
      shadow_arg.set_name(arg_name);
      if (arg.HasMember("s_f")) {
        shadow_arg.set_s_f(Json::GetFloat(arg, "s_f", 0));
      } else if (arg.HasMember("s_i")) {
        shadow_arg.set_s_i(Json::GetInt(arg, "s_i", 0));
      } else if (arg.HasMember("s_s")) {
        shadow_arg.set_s_s(Json::GetString(arg, "s_s", "None"));
      } else if (arg.HasMember("v_f")) {
        for (const auto &v : Json::GetVecFloat(arg, "v_f")) {
          shadow_arg.add_v_f(v);
        }
      } else if (arg.HasMember("v_i")) {
        for (const auto &v : Json::GetVecInt(arg, "v_i")) {
          shadow_arg.add_v_i(v);
        }
      } else if (arg.HasMember("v_s")) {
        for (const auto &v : Json::GetVecString(arg, "v_s")) {
          shadow_arg.add_v_s(v);
        }
      } else {
        LOG(FATAL) << "Unsupported argument: " << arg_name;
      }
      net->add_arg(shadow_arg);
    }
  }

  if (document.HasMember("blob")) {
    const auto &json_blobs = document["blob"];
    CHECK(json_blobs.IsArray());
    for (int n = 0; n < json_blobs.Size(); ++n) {
      const auto &blob = json_blobs[n];
      shadow::Blob shadow_blob;
      shadow_blob.set_name(Json::GetString(blob, "name", "None"));
      shadow_blob.set_type(Json::GetString(blob, "type", "float"));
      if (blob.HasMember("shape")) {
        const auto &shape = Json::GetVecInt(blob, "shape");
        for (const auto &dim : shape) {
          shadow_blob.add_shape(dim);
        }
      }
      net->add_blob(shadow_blob);
    }
  }

  for (int o = 0; o < json_ops.Size(); ++o) {
    const auto &json_op = json_ops[o];
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

#else
struct Arg {
  std::string type;
  float s_f;
  int s_i;
  std::string s_s;
  std::vector<float> v_f;
  std::vector<int> v_i;
  std::vector<std::string> v_s;
};

const std::map<std::string, Arg> ParseArgument(
    const std::string &argument_str) {
  std::map<std::string, Arg> arg_map;
  for (const auto &argument : Util::tokenize(argument_str, ";")) {
    auto arg_trim = Util::trim(argument);
    arg_trim = arg_trim.substr(1, arg_trim.size() - 2);
    auto arg_part = Util::tokenize(arg_trim, ",");
    CHECK_EQ(arg_part.size(), 3);
    for (auto &part : arg_part) {
      part = Util::trim(part);
    }
    const auto &arg_name = arg_part[0];
    const auto &arg_type = arg_part[1];
    const auto &arg_value = arg_part[2].substr(1, arg_part[2].size() - 2);
    Arg arg{};
    arg.type = arg_type;
    if (arg_type == "s_f") {
      arg.s_f = Util::stof(arg_value);
    } else if (arg_type == "s_i") {
      arg.s_i = Util::stoi(arg_value);
    } else if (arg_type == "s_s") {
      arg.s_s = arg_value;
    } else if (arg_type == "v_f") {
      for (const auto &v : Util::tokenize(arg_value, "#")) {
        arg.v_f.push_back(Util::stof(Util::trim(v)));
      }
    } else if (arg_type == "v_i") {
      for (const auto &v : Util::tokenize(arg_value, "#")) {
        arg.v_i.push_back(Util::stoi(Util::trim(v)));
      }
    } else if (arg_type == "v_s") {
      for (const auto &v : Util::tokenize(arg_value, "#")) {
        arg.v_s.push_back(Util::trim(v));
      }
    } else {
      LOG(FATAL) << "Unsupported argument type " << arg_type;
    }
    arg_map[arg_name] = arg;
  }
  return arg_map;
}

const std::map<std::string, Arg> ParseCommon(
    const std::vector<std::string> &params, shadow::OpParam *op) {
  op->set_type(params[0]);
  op->set_name(params[1]);
  const auto &counts = Util::tokenize(params[2], " ");
  CHECK_EQ(counts.size(), 3);
  if (Util::stoi(counts[0]) > 0) {
    for (const auto &bottom : Util::tokenize(params[3], ";")) {
      op->add_bottom(Util::trim(bottom));
    }
  }
  if (Util::stoi(counts[1]) > 0) {
    for (const auto &top : Util::tokenize(params[4], ";")) {
      op->add_top(Util::trim(top));
    }
  }
  std::map<std::string, Arg> arg_map;
  if (Util::stoi(counts[2]) > 0) {
    arg_map = ParseArgument(params[5]);
  }
  return arg_map;
}

const shadow::OpParam ParseActivate(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int type = 1, channel_shared = false;
  if (argument.count("type")) {
    type = argument.at("type").s_i;
  }
  if (argument.count("channel_shared")) {
    channel_shared = argument.at("channel_shared").s_i;
  }

  set_s_i(&shadow_op, "type", type);
  set_s_i(&shadow_op, "channel_shared", channel_shared);

  return shadow_op;
}

const shadow::OpParam ParseBatchNorm(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int use_global_stats = true;
  if (argument.count("use_global_stats")) {
    use_global_stats = argument.at("use_global_stats").s_i;
  }

  set_s_i(&shadow_op, "use_global_stats", use_global_stats);

  return shadow_op;
}

const shadow::OpParam ParseBinary(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int operation = -1;
  float scalar = 0;
  bool has_scalar = false;
  if (argument.count("operation")) {
    operation = argument.at("operation").s_i;
  }
  if (argument.count("scalar")) {
    scalar = argument.at("scalar").s_f;
    has_scalar = true;
  }

  set_s_i(&shadow_op, "operation", operation);
  if (has_scalar) {
    set_s_f(&shadow_op, "scalar", scalar);
  }

  return shadow_op;
}

const shadow::OpParam ParseConcat(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int axis = 1;
  if (argument.count("axis")) {
    axis = argument.at("axis").s_i;
  }

  set_s_i(&shadow_op, "axis", axis);

  return shadow_op;
}

const shadow::OpParam ParseConnected(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int num_output = -1, bias_term = true, transpose = true;
  if (argument.count("num_output")) {
    num_output = argument.at("num_output").s_i;
  }
  if (argument.count("bias_term")) {
    bias_term = argument.at("bias_term").s_i;
  }
  if (argument.count("transpose")) {
    transpose = argument.at("transpose").s_i;
  }

  CHECK_GT(num_output, 0);
  set_s_i(&shadow_op, "num_output", num_output);
  set_s_i(&shadow_op, "bias_term", bias_term);
  set_s_i(&shadow_op, "transpose", transpose);

  return shadow_op;
}

const shadow::OpParam ParseConv(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int num_output = -1, kernel_size = -1, stride = 1, pad = 0, dilation = 1,
      group = 1, bias_term = true, type = -1;
  if (argument.count("num_output")) {
    num_output = argument.at("num_output").s_i;
  }
  if (argument.count("kernel_size")) {
    kernel_size = argument.at("kernel_size").s_i;
  }
  if (argument.count("stride")) {
    stride = argument.at("stride").s_i;
  }
  if (argument.count("pad")) {
    pad = argument.at("pad").s_i;
  }
  if (argument.count("dilation")) {
    dilation = argument.at("dilation").s_i;
  }
  if (argument.count("group")) {
    group = argument.at("group").s_i;
  }
  if (argument.count("bias_term")) {
    bias_term = argument.at("bias_term").s_i;
  }
  if (argument.count("type")) {
    type = argument.at("type").s_i;
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
  set_s_i(&shadow_op, "type", type);

  return shadow_op;
}

const shadow::OpParam ParseEltwise(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int operation = 1;
  std::vector<float> coeffs;
  if (argument.count("operation")) {
    operation = argument.at("operation").s_i;
  }
  if (argument.count("coeff")) {
    coeffs = argument.at("coeff").v_f;
  }

  set_s_i(&shadow_op, "operation", operation);
  set_v_f(&shadow_op, "coeff", coeffs);

  return shadow_op;
}

const shadow::OpParam ParseFlatten(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int axis = 1, end_axis = -1;
  if (argument.count("axis")) {
    axis = argument.at("axis").s_i;
  }
  if (argument.count("end_axis")) {
    end_axis = argument.at("end_axis").s_i;
  }

  set_s_i(&shadow_op, "axis", axis);
  set_s_i(&shadow_op, "end_axis", end_axis);

  return shadow_op;
}

const shadow::OpParam ParseInput(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  for (const auto &arg : argument) {
    set_v_i(&shadow_op, arg.first, arg.second.v_i);
  }

  return shadow_op;
}

const shadow::OpParam ParseLRN(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int norm_region = 0, local_size = 5;
  float alpha = 1, beta = 0.75, k = 1;
  if (argument.count("local_size")) {
    local_size = argument.at("local_size").s_i;
  }
  if (argument.count("alpha")) {
    alpha = argument.at("alpha").s_f;
  }
  if (argument.count("beta")) {
    beta = argument.at("beta").s_f;
  }
  if (argument.count("k")) {
    k = argument.at("k").s_f;
  }
  if (argument.count("norm_region")) {
    norm_region = argument.at("norm_region").s_i;
  }

  set_s_i(&shadow_op, "local_size", local_size);
  set_s_f(&shadow_op, "alpha", alpha);
  set_s_f(&shadow_op, "beta", beta);
  set_s_f(&shadow_op, "k", k);
  set_s_i(&shadow_op, "norm_region", norm_region);

  return shadow_op;
}

const shadow::OpParam ParseNormalize(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int across_spatial = true, channel_shared = true;
  std::vector<float> scale;
  if (argument.count("across_spatial")) {
    across_spatial = argument.at("across_spatial").s_i;
  }
  if (argument.count("channel_shared")) {
    channel_shared = argument.at("channel_shared").s_i;
  }
  if (argument.count("scale")) {
    scale = argument.at("scale").v_f;
  }

  set_s_i(&shadow_op, "across_spatial", across_spatial);
  set_s_i(&shadow_op, "channel_shared", channel_shared);
  set_v_f(&shadow_op, "scale", scale);

  return shadow_op;
}

const shadow::OpParam ParsePermute(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  std::vector<int> order;
  if (argument.count("order")) {
    order = argument.at("order").v_i;
  }

  set_v_i(&shadow_op, "order", order);

  return shadow_op;
}

const shadow::OpParam ParsePooling(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int pool = 0, global_pooling = false;
  std::vector<int> kernel_size, stride, pad;
  if (argument.count("pool")) {
    pool = argument.at("pool").s_i;
  }
  if (argument.count("kernel_size")) {
    kernel_size = argument.at("kernel_size").v_i;
  }
  if (argument.count("stride")) {
    stride = argument.at("stride").v_i;
  }
  if (argument.count("pad")) {
    pad = argument.at("pad").v_i;
  }
  if (argument.count("global_pooling")) {
    global_pooling = argument.at("global_pooling").s_i;
  }

  set_s_i(&shadow_op, "pool", pool);
  set_v_i(&shadow_op, "kernel_size", kernel_size);
  set_v_i(&shadow_op, "stride", stride);
  set_v_i(&shadow_op, "pad", pad);
  set_s_i(&shadow_op, "global_pooling", global_pooling);

  return shadow_op;
}

const shadow::OpParam ParsePriorBox(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  std::vector<float> min_size, max_size, aspect_ratio, variance;
  int flip = true, clip = false;
  float step = -1, offset = 0.5;
  if (argument.count("min_size")) {
    min_size = argument.at("min_size").v_f;
  }
  if (argument.count("max_size")) {
    max_size = argument.at("max_size").v_f;
  }
  if (argument.count("aspect_ratio")) {
    aspect_ratio = argument.at("aspect_ratio").v_f;
  }
  if (argument.count("flip")) {
    flip = argument.at("flip").s_i;
  }
  if (argument.count("clip")) {
    clip = argument.at("clip").s_i;
  }
  if (argument.count("variance")) {
    variance = argument.at("variance").v_f;
  }
  if (argument.count("step")) {
    step = argument.at("step").s_f;
  }
  if (argument.count("offset")) {
    offset = argument.at("offset").s_f;
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

const shadow::OpParam ParseReorg(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int stride = 2;
  if (argument.count("stride")) {
    stride = argument.at("stride").s_i;
  }

  set_s_i(&shadow_op, "stride", stride);

  return shadow_op;
}

const shadow::OpParam ParseReshape(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  std::vector<int> shape;
  int axis = 0, num_axes = -1;
  if (argument.count("shape")) {
    shape = argument.at("shape").v_i;
  }
  if (argument.count("axis")) {
    axis = argument.at("axis").s_i;
  }
  if (argument.count("num_axes")) {
    num_axes = argument.at("num_axes").s_i;
  }

  set_v_i(&shadow_op, "shape", shape);
  set_s_i(&shadow_op, "axis", axis);
  set_s_i(&shadow_op, "num_axes", num_axes);

  return shadow_op;
}

const shadow::OpParam ParseScale(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  std::vector<float> scale_value, bias_value;
  int axis = 1, has_scale = true, has_bias = true;
  if (argument.count("axis")) {
    axis = argument.at("axis").s_i;
  }
  if (argument.count("has_scale")) {
    has_scale = argument.at("has_scale").s_i;
  }
  if (argument.count("has_bias")) {
    has_bias = argument.at("has_bias").s_i;
  }
  if (argument.count("scale_value")) {
    scale_value = argument.at("scale_value").v_f;
  }
  if (argument.count("bias_value")) {
    bias_value = argument.at("bias_value").v_f;
  }

  set_s_i(&shadow_op, "axis", axis);
  set_s_i(&shadow_op, "has_scale", has_scale);
  set_s_i(&shadow_op, "has_bias", has_bias);
  set_v_f(&shadow_op, "scale_value", scale_value);
  set_v_f(&shadow_op, "bias_value", bias_value);

  return shadow_op;
}

const shadow::OpParam ParseSlice(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int axis = 1;
  std::vector<int> slice_point;
  if (argument.count("axis")) {
    axis = argument.at("axis").s_i;
  }
  if (argument.count("slice_point")) {
    slice_point = argument.at("slice_point").v_i;
  }

  set_s_i(&shadow_op, "axis", axis);
  set_v_i(&shadow_op, "slice_point", slice_point);

  return shadow_op;
}

const shadow::OpParam ParseSoftmax(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int axis = 1;
  if (argument.count("axis")) {
    axis = argument.at("axis").s_i;
  }

  set_s_i(&shadow_op, "axis", axis);

  return shadow_op;
}

const shadow::OpParam ParseStack(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int axis = 0;
  if (argument.count("axis")) {
    axis = argument.at("axis").s_i;
  }

  set_s_i(&shadow_op, "axis", axis);

  return shadow_op;
}

const shadow::OpParam ParseUnary(const std::vector<std::string> &params) {
  shadow::OpParam shadow_op;

  const auto &argument = ParseCommon(params, &shadow_op);

  int operation = -1;
  if (argument.count("operation")) {
    operation = argument.at("operation").s_i;
  }

  set_s_i(&shadow_op, "operation", operation);

  return shadow_op;
}

using ParseFunc =
    std::function<const shadow::OpParam(const std::vector<std::string> &)>;

static const std::map<std::string, ParseFunc> parse_func_map{
    {"Activate", ParseActivate},
    {"BatchNorm", ParseBatchNorm},
    {"Binary", ParseBinary},
    {"Concat", ParseConcat},
    {"Connected", ParseConnected},
    {"Conv", ParseConv},
    {"Deconv", ParseConv},
    {"Eltwise", ParseEltwise},
    {"Flatten", ParseFlatten},
    {"Input", ParseInput},
    {"LRN", ParseLRN},
    {"Normalize", ParseNormalize},
    {"Permute", ParsePermute},
    {"Pooling", ParsePooling},
    {"PriorBox", ParsePriorBox},
    {"Reorg", ParseReorg},
    {"Reshape", ParseReshape},
    {"Scale", ParseScale},
    {"Slice", ParseSlice},
    {"Softmax", ParseSoftmax},
    {"Stack", ParseStack},
    {"Unary", ParseUnary}};

void ParseNet(const std::string &proto_text, shadow::NetParam *net) {
  const auto &marks = Util::tokenize(proto_text, "\n");
  CHECK_GT(marks.size(), 0);

  net->set_name(marks[0].substr(1, marks[0].size() - 2));

  const auto &argument = marks[1].substr(1, marks[1].size() - 2);
  for (const auto &arg : ParseArgument(argument)) {
    shadow::Argument shadow_arg;
    shadow_arg.set_name(arg.first);
    if (arg.second.type == "s_f") {
      shadow_arg.set_s_f(arg.second.s_f);
    } else if (arg.second.type == "s_i") {
      shadow_arg.set_s_i(arg.second.s_i);
    } else if (arg.second.type == "s_s") {
      shadow_arg.set_s_s(arg.second.s_s);
    } else if (arg.second.type == "v_f") {
      for (const auto &v : arg.second.v_f) {
        shadow_arg.add_v_f(v);
      }
    } else if (arg.second.type == "v_i") {
      for (const auto &v : arg.second.v_i) {
        shadow_arg.add_v_i(v);
      }
    } else if (arg.second.type == "v_s") {
      for (const auto &v : arg.second.v_s) {
        shadow_arg.add_v_s(v);
      }
    }
    net->add_arg(shadow_arg);
  }

  const auto &blobs = marks[2].substr(1, marks[2].size() - 2);
  for (const auto &blob : Util::tokenize(blobs, ";")) {
    auto blob_trim = Util::trim(blob);
    blob_trim = blob_trim.substr(1, blob_trim.size() - 2);
    auto blob_part = Util::tokenize(blob_trim, ",");
    CHECK_EQ(blob_part.size(), 3);
    for (auto &part : blob_part) {
      part = Util::trim(part);
    }
    shadow::Blob shadow_blob;
    shadow_blob.set_name(blob_part[0]);
    shadow_blob.set_type(blob_part[1]);
    const auto &blob_shape = blob_part[2].substr(1, blob_part[2].size() - 2);
    for (const auto &dim : Util::tokenize(blob_shape, "#")) {
      shadow_blob.add_shape(Util::stoi(Util::trim(dim)));
    }
    net->add_blob(shadow_blob);
  }

  for (int i = 3; i < marks.size(); ++i) {
    auto params = Util::tokenize(marks[i], "|");
    CHECK_EQ(params.size(), 6);
    for (auto &param : params) {
      param = Util::trim(param);
      param = param.substr(1, param.size() - 2);
    }

    const auto &op_type = params[0];
    const auto &op_name = params[1];

    bool find_parser = false;
    for (const auto &it : parse_func_map) {
      if (op_type.find(it.first) != std::string::npos) {
        net->add_op(it.second(params));
        find_parser = true;
        break;
      }
    }
    CHECK(find_parser) << "Error when parsing op: " << op_name
                       << ", op type: " << op_type << " is not recognized!";
  }
}
#endif

#endif

}  // namespace Parser

}  // namespace Shadow
