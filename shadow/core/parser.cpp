#include "parser.hpp"
#include "util/log.hpp"

namespace Shadow {

#if !defined(USE_Protobuf)
namespace Parser {

void ParseNet(const std::string &proto_text, shadow::NetParam *net) {
  const auto &document = Json::GetDocument(proto_text);

  const auto &net_name = Json::GetString(document, "name", "");
  const auto &json_ops = Json::GetValue(document, "op");

  net->set_name(net_name);

  for (int i = 0; i < json_ops.Size(); ++i) {
    const auto &json_op = json_ops[i];
    const auto &op_name = Json::GetString(json_op, "name", "");
    const auto &op_type = Json::GetString(json_op, "type", "");
    if (!op_type.compare("Activate")) {
      net->add_op(ParseActivate(json_op));
    } else if (!op_type.compare("BatchNorm")) {
      net->add_op(ParseBatchNorm(json_op));
    } else if (!op_type.compare("Bias")) {
      net->add_op(ParseBias(json_op));
    } else if (!op_type.compare("Concat")) {
      net->add_op(ParseConcat(json_op));
    } else if (!op_type.compare("Connected")) {
      net->add_op(ParseConnected(json_op));
    } else if (!op_type.compare("Convolution")) {
      net->add_op(ParseConvolution(json_op));
    } else if (!op_type.compare("Data")) {
      net->add_op(ParseData(json_op));
    } else if (!op_type.compare("Eltwise")) {
      net->add_op(ParseEltwise(json_op));
    } else if (!op_type.compare("Flatten")) {
      net->add_op(ParseFlatten(json_op));
    } else if (!op_type.compare("LRN")) {
      net->add_op(ParseLRN(json_op));
    } else if (!op_type.compare("Normalize")) {
      net->add_op(ParseNormalize(json_op));
    } else if (!op_type.compare("Permute")) {
      net->add_op(ParsePermute(json_op));
    } else if (!op_type.compare("Pooling")) {
      net->add_op(ParsePooling(json_op));
    } else if (!op_type.compare("PriorBox")) {
      net->add_op(ParsePriorBox(json_op));
    } else if (!op_type.compare("Reorg")) {
      net->add_op(ParseReorg(json_op));
    } else if (!op_type.compare("Reshape")) {
      net->add_op(ParseReshape(json_op));
    } else if (!op_type.compare("Scale")) {
      net->add_op(ParseScale(json_op));
    } else if (!op_type.compare("Softmax")) {
      net->add_op(ParseSoftmax(json_op));
    } else {
      LOG(FATAL) << "Error when parsing op: " << op_name
                 << ", op type: " << op_type << " is not recognized!";
    }
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
      if (!arg_name.compare("type")) {
        type = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("channel_shared")) {
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
      if (!arg_name.compare("use_global_stats")) {
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
      if (!arg_name.compare("axis")) {
        axis = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("num_axes")) {
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
      if (!arg_name.compare("axis")) {
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
      if (!arg_name.compare("num_output")) {
        num_output = Json::GetInt(arg, "s_i", -1);
      } else if (!arg_name.compare("bias_term")) {
        bias_term = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("transpose")) {
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

const shadow::OpParam ParseConvolution(const JValue &root) {
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
      if (!arg_name.compare("num_output")) {
        num_output = Json::GetInt(arg, "s_i", -1);
      } else if (!arg_name.compare("kernel_size")) {
        kernel_size = Json::GetInt(arg, "s_i", -1);
      } else if (!arg_name.compare("stride")) {
        stride = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("pad")) {
        pad = Json::GetInt(arg, "s_i", 0);
      } else if (!arg_name.compare("dilation")) {
        dilation = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("group")) {
        group = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("bias_term")) {
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
      if (!arg_name.compare("data_shape")) {
        data_shape = Json::GetVecInt(arg, "v_i");
      } else if (!arg_name.compare("scale")) {
        scale = Json::GetFloat(arg, "s_f", 1);
      } else if (!arg_name.compare("mean_value")) {
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
      if (!arg_name.compare("operation")) {
        operation = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("coeff")) {
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
      if (!arg_name.compare("axis")) {
        axis = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("end_axis")) {
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
      if (!arg_name.compare("local_size")) {
        local_size = Json::GetInt(arg, "s_i", 5);
      } else if (!arg_name.compare("alpha")) {
        alpha = Json::GetFloat(arg, "s_f", -1.f);
      } else if (!arg_name.compare("beta")) {
        beta = Json::GetFloat(arg, "s_f", 0.75f);
      } else if (!arg_name.compare("k")) {
        k = Json::GetFloat(arg, "s_f", 1);
      } else if (!arg_name.compare("norm_region")) {
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
      if (!arg_name.compare("across_spatial")) {
        across_spatial = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("channel_shared")) {
        channel_shared = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("scale")) {
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
      if (!arg_name.compare("order")) {
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
      if (!arg_name.compare("pool")) {
        pool = Json::GetInt(arg, "s_i", 0);
      } else if (!arg_name.compare("kernel_size")) {
        kernel_size = Json::GetInt(arg, "s_i", -1);
      } else if (!arg_name.compare("stride")) {
        stride = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("pad")) {
        pad = Json::GetInt(arg, "s_i", 0);
      } else if (!arg_name.compare("global_pooling")) {
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
      if (!arg_name.compare("min_size")) {
        min_size = Json::GetVecFloat(arg, "v_f");
      } else if (!arg_name.compare("max_size")) {
        max_size = Json::GetVecFloat(arg, "v_f");
      } else if (!arg_name.compare("aspect_ratio")) {
        aspect_ratio = Json::GetVecFloat(arg, "v_f");
      } else if (!arg_name.compare("flip")) {
        flip = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("clip")) {
        clip = Json::GetInt(arg, "s_i", 0);
      } else if (!arg_name.compare("variance")) {
        variance = Json::GetVecFloat(arg, "v_f");
      } else if (!arg_name.compare("step")) {
        step = Json::GetFloat(arg, "s_f", -1);
      } else if (!arg_name.compare("offset")) {
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
  set_s_f(&shadow_op, "step", step);
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
      if (!arg_name.compare("stride")) {
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
      if (!arg_name.compare("shape")) {
        shape = Json::GetVecInt(arg, "v_i");
      } else if (!arg_name.compare("axis")) {
        axis = Json::GetInt(arg, "s_i", 0);
      } else if (!arg_name.compare("num_axes")) {
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
      if (!arg_name.compare("axis")) {
        axis = Json::GetInt(arg, "s_i", 1);
      } else if (!arg_name.compare("num_axes")) {
        num_axes = Json::GetInt(arg, "s_i", 0);
      } else if (!arg_name.compare("bias_term")) {
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
      if (!arg_name.compare("axis")) {
        axis = Json::GetInt(arg, "s_i", 1);
      }
    }
  }

  set_s_i(&shadow_op, "axis", axis);

  return shadow_op;
}

}  // namespace Parser
#endif

}  // namespace Shadow
