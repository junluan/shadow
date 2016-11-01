#ifndef SHADOW_TOOLS_CAFFE2SHADOW_HPP
#define SHADOW_TOOLS_CAFFE2SHADOW_HPP

#include "shadow/util/io.hpp"
#include "shadow/util/util.hpp"

#include "caffe.pb.h"
#include "shadow.pb.h"

namespace Caffe2Shadow {

void ConvertCommon(const caffe::NetParameter& caffe_model,
                   const caffe::LayerParameter& caffe_layer,
                   shadow::LayerParameter* shadow_layer) {
  const std::string& layer_name = caffe_layer.name();
  shadow_layer->set_name(layer_name);
  for (int n = 0; n < caffe_layer.top_size(); ++n) {
    shadow_layer->add_top(caffe_layer.top(n));
  }
  for (int n = 0; n < caffe_layer.bottom_size(); ++n) {
    shadow_layer->add_bottom(caffe_layer.bottom(n));
  }
  for (int l = 0; l < caffe_model.layer_size(); ++l) {
    const caffe::LayerParameter& layer = caffe_model.layer(l);
    if (!layer_name.compare(layer.name())) {
      for (int n = 0; n < layer.blobs_size(); ++n) {
        const caffe::BlobProto& caffe_blob = layer.blobs(n);
        shadow::Blob* shadow_blob = shadow_layer->add_blobs();
        int data_size = caffe_blob.data_size();
        shadow_blob->set_count(data_size);
        for (int i = 0; i < data_size; ++i) {
          shadow_blob->add_data(caffe_blob.data(i));
        }
      }
      break;
    }
  }
}

void ConvertData(const caffe::LayerParameter& data_model,
                 const caffe::LayerParameter& data_deploy,
                 shadow::NetParameter* shadow_net) {
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_name(data_deploy.name());
  shadow_layer->set_type("Data");
  for (int n = 0; n < data_deploy.top_size(); ++n) {
    shadow_layer->add_top(data_deploy.top(n));
  }

  shadow::DataParameter* shadow_param = shadow_layer->mutable_data_param();
  if (data_model.has_transform_param()) {
    const caffe::TransformationParameter& caffe_param =
        data_model.transform_param();
    if (caffe_param.has_scale()) {
      shadow_param->set_scale(caffe_param.scale());
    }
    if (caffe_param.mean_value_size() > 0) {
      for (int n = 0; n < caffe_param.mean_value_size(); ++n) {
        shadow_param->add_mean_value(caffe_param.mean_value(n));
      }
    }
  }
}

void ConvertActivate(const caffe::NetParameter& caffe_model,
                     const caffe::LayerParameter& caffe_layer,
                     shadow::NetParameter* shadow_net) {
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Activate");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::ActivateParameter* shadow_param =
      shadow_layer->mutable_activate_param();
  if (!caffe_layer.type().compare("ReLU")) {
    shadow_param->set_type(shadow::ActivateParameter_ActivateType_Relu);
  }
}

void ConvertConcat(const caffe::NetParameter& caffe_model,
                   const caffe::LayerParameter& caffe_layer,
                   shadow::NetParameter* shadow_net) {
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Concat");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::ConcatParameter* shadow_param = shadow_layer->mutable_concat_param();
  if (caffe_layer.has_concat_param()) {
    const caffe::ConcatParameter& caffe_param = caffe_layer.concat_param();
    if (caffe_param.has_axis()) {
      shadow_param->set_axis(caffe_param.axis());
    }
  }
}

void ConvertConnected(const caffe::NetParameter& caffe_model,
                      const caffe::LayerParameter& caffe_layer,
                      shadow::NetParameter* shadow_net) {
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Connected");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::ConnectedParameter* shadow_param =
      shadow_layer->mutable_connected_param();
  if (caffe_layer.has_inner_product_param()) {
    const caffe::InnerProductParameter& caffe_param =
        caffe_layer.inner_product_param();
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
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Convolution");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::ConvolutionParameter* shadow_param =
      shadow_layer->mutable_convolution_param();
  if (caffe_layer.has_convolution_param()) {
    const caffe::ConvolutionParameter& caffe_param =
        caffe_layer.convolution_param();
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
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Flatten");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::FlattenParameter* shadow_param =
      shadow_layer->mutable_flatten_param();
  if (caffe_layer.has_flatten_param()) {
    const caffe::FlattenParameter& caffe_param = caffe_layer.flatten_param();
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
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Normalize");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::NormalizeParameter* shadow_param =
      shadow_layer->mutable_normalize_param();
  if (caffe_layer.has_norm_param()) {
    const caffe::NormalizeParameter& caffe_param = caffe_layer.norm_param();
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
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Permute");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::PermuteParameter* shadow_param =
      shadow_layer->mutable_permute_param();
  if (caffe_layer.has_permute_param()) {
    const caffe::PermuteParameter& caffe_param = caffe_layer.permute_param();
    for (int i = 0; i < caffe_param.order_size(); ++i) {
      shadow_param->add_order(caffe_param.order(i));
    }
  }
}

void ConvertPooling(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Pooling");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::PoolingParameter* shadow_param =
      shadow_layer->mutable_pooling_param();
  if (caffe_layer.has_pooling_param()) {
    const caffe::PoolingParameter& caffe_param = caffe_layer.pooling_param();
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
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("PriorBox");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::PriorBoxParameter* shadow_param =
      shadow_layer->mutable_prior_box_param();
  if (caffe_layer.has_prior_box_param()) {
    const caffe::PriorBoxParameter& caffe_param = caffe_layer.prior_box_param();
    if (caffe_param.has_min_size()) {
      shadow_param->set_min_size(caffe_param.min_size());
    }
    if (caffe_param.has_max_size()) {
      shadow_param->set_max_size(caffe_param.max_size());
    }
    for (int i = 0; i < caffe_param.aspect_ratio_size(); ++i) {
      shadow_param->add_aspect_ratio(caffe_param.aspect_ratio(i));
    }
    if (caffe_param.has_flip()) {
      shadow_param->set_flip(caffe_param.flip());
    }
    if (caffe_param.has_clip()) {
      shadow_param->set_clip(caffe_param.clip());
    }
    for (int i = 0; i < caffe_param.variance_size(); ++i) {
      shadow_param->add_variance(caffe_param.variance(i));
    }
  }
}

void ConvertReshape(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Reshape");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::ReshapeParameter* shadow_param =
      shadow_layer->mutable_reshape_param();
  if (caffe_layer.has_reshape_param()) {
    const caffe::ReshapeParameter& caffe_param = caffe_layer.reshape_param();
    for (int i = 0; i < caffe_param.shape().dim_size(); ++i) {
      shadow_param->mutable_shape()->add_dim(caffe_param.shape().dim(i));
    }
    if (caffe_param.has_axis() && caffe_param.axis() != 0) {
      shadow_param->set_axis(caffe_param.axis());
    }
    if (caffe_param.has_num_axes() && caffe_param.num_axes() != -1) {
      shadow_param->set_num_axes(caffe_param.num_axes());
    }
  }
}

void ConvertSoftmax(const caffe::NetParameter& caffe_model,
                    const caffe::LayerParameter& caffe_layer,
                    shadow::NetParameter* shadow_net) {
  shadow::LayerParameter* shadow_layer = shadow_net->add_layer();
  shadow_layer->set_type("Softmax");
  ConvertCommon(caffe_model, caffe_layer, shadow_layer);

  shadow::SoftmaxParameter* shadow_param =
      shadow_layer->mutable_softmax_param();
  if (caffe_layer.has_softmax_param()) {
    const caffe::SoftmaxParameter& caffe_param = caffe_layer.softmax_param();
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
    const caffe::LayerParameter& caffe_layer = caffe_deploy.layer(l);
    if (!caffe_layer.type().compare("ReLU")) {
      ConvertActivate(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("Concat")) {
      ConvertConcat(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("InnerProduct")) {
      ConvertConnected(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("Convolution")) {
      ConvertConv(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("Flatten")) {
      ConvertFlatten(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("Normalize")) {
      ConvertNormalize(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("Permute")) {
      ConvertPermute(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("Pooling")) {
      ConvertPooling(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("PriorBox")) {
      ConvertPriorBox(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("Reshape")) {
      ConvertReshape(caffe_model, caffe_layer, shadow_net);
    } else if (!caffe_layer.type().compare("Softmax")) {
      ConvertSoftmax(caffe_model, caffe_layer, shadow_net);
    } else {
      Warning("Layer type: " + caffe_layer.type() + " is not recognized!");
    }
  }
}

}  // namespace Caffe2Shadow

namespace Caffe2Shadow {

using google::protobuf::Message;

void WriteProtoToCPP(const shadow::NetParameter& shadow_net,
                     const std::string& root, const std::string& model_name) {
  shadow::NetParameter net(shadow_net);
  for (int l = 0; l < net.layer_size(); ++l) {
    shadow::LayerParameter* layer_param = net.mutable_layer(l);
    if (layer_param->blobs_size() == 0) continue;
    for (int n = 0; n < layer_param->blobs_size(); ++n) {
      layer_param->mutable_blobs(n)->clear_data();
    }
  }

  std::string filename = root + "/" + model_name + ".cpp";
  std::ofstream file(filename);
  std::string proto_str;
  IO::WriteProtoToText(net, &proto_str);

  file << "#include \"" << model_name << ".hpp\"\n\n";

  file << "const std::string Model::model_ = \nR\"(\n";
  file << proto_str;
  file << ")\";\n";

  file.close();
}

void WriteDefinesToHPP(const shadow::NetParameter& shadow_net,
                       const std::string& root, const std::string& model_name) {
  VecInt weight_counts;
  VecString weight_names;
  for (int l = 0; l < shadow_net.layer_size(); ++l) {
    const shadow::LayerParameter& layer_param = shadow_net.layer(l);
    if (layer_param.blobs_size() == 0) continue;
    int count = 0;
    for (int n = 0; n < layer_param.blobs_size(); ++n) {
      count += layer_param.blobs(n).data_size();
    }
    weight_counts.push_back(count);
    weight_names.push_back(model_name + "_" + layer_param.name());
  }

  std::ofstream file(root + "/" + model_name + ".hpp");

  file << "#ifndef SHADOW_MODEL_HPP\n"
          "#define SHADOW_MODEL_HPP\n\n";

  file << "#include <cstring>\n"
       << "#include <string>\n\n";

  file << "static constexpr int counts_[] = "
       << Util::format_vector(weight_counts, ", ", "{", "}") << ";\n\n";

  file << "class Model {\n"
          " public:\n"
          "  static const std::string model() { return model_; }\n\n";

  file << "  static const float *weights(int n) {\n"
          "    switch (n) {\n";
  for (int i = 0; i < weight_names.size(); ++i) {
    file << "      case " << i << ":\n"
         << "        return " << weight_names[i] << "_;\n";
  }
  file << "      default:\n"
          "        return nullptr;\n"
          "    }\n"
          "  }\n"
          "  static void get_weights(float *weights_data) {\n"
          "    for (int n = 0; n < num(); ++n) {\n"
          "      memcpy(weights_data, weights(n), count(n) * sizeof(float));\n"
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

  file << " private:\n"
          "  static const std::string model_;\n\n";

  for (int i = 0; i < weight_names.size(); ++i) {
    file << "  static const float *" << weight_names[i] << "_;\n";
  }
  file << "};\n\n";

  file << "#endif  // SHADOW_MODEL_HPP\n";

  file.close();
}

void WriteWeightsToCPP(const shadow::NetParameter& shadow_net,
                       const std::string& root, const std::string& model_name) {
  for (int l = 0; l < shadow_net.layer_size(); ++l) {
    const shadow::LayerParameter& layer_param = shadow_net.layer(l);
    if (layer_param.blobs_size() == 0) continue;

    std::string weight_name = model_name + "_" + layer_param.name();
    std::string filename = root + "/" + weight_name + ".cpp";
    std::ofstream file(filename);

    file << "#include \"" << model_name << ".hpp\"\n\n";
    file << "const float " << weight_name << "_weights[] = {\n";

    int count = 0, num_of_line = 10;
    for (int n = 0; n < layer_param.blobs_size(); ++n) {
      int data_size = layer_param.blobs(n).data_size();
      for (int i = 0; i < data_size; ++i) {
        if (count > 0) {
          file << ",";
        }
        if (count > 0 && count % num_of_line == 0) {
          file << "\n";
        }
        file << layer_param.blobs(n).data(i);
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
  WriteDefinesToHPP(shadow_net, root, model_name);
  WriteWeightsToCPP(shadow_net, root, model_name);
  WriteProtoToCPP(shadow_net, root, model_name);
}

void WriteProtoToBinary(const Message& proto, const std::string& root,
                        const std::string& model_name) {
  std::string filename = root + "/" + model_name + ".shadowmodel";
  IO::WriteProtoToBinaryFile(proto, filename);
}

}  // namespace Caffe2Shadow

#endif  // SHADOW_TOOLS_CAFFE2SHADOW_HPP
