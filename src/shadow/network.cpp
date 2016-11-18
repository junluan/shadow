#include "shadow/network.hpp"
#include "shadow/util/io.hpp"

#include "shadow/layers/activate_layer.hpp"
#include "shadow/layers/concat_layer.hpp"
#include "shadow/layers/connected_layer.hpp"
#include "shadow/layers/convolution_layer.hpp"
#include "shadow/layers/data_layer.hpp"
#include "shadow/layers/flatten_layer.hpp"
#include "shadow/layers/normalize_layer.hpp"
#include "shadow/layers/permute_layer.hpp"
#include "shadow/layers/pooling_layer.hpp"
#include "shadow/layers/prior_box_layer.hpp"
#include "shadow/layers/reshape_layer.hpp"
#include "shadow/layers/softmax_layer.hpp"

void Network::LoadModel(const std::string &proto_bin, int batch) {
  LoadProtoBin(proto_bin, &net_param_);
  Reshape(batch);
}

void Network::LoadModel(const std::string &proto_str, const float *weights_data,
                        int batch) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Reshape(batch);
  CopyWeights(weights_data);
}

void Network::LoadModel(const std::string &proto_str,
                        const std::vector<const float *> &weights, int batch) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Reshape(batch);
  CopyWeights(weights);
}

void Network::SaveModel(const std::string &proto_bin) {
  for (int l = 0; l < layers_.size(); ++l) {
    net_param_.mutable_layer(l)->clear_blobs();
    const Layer *layer = layers_[l];
    for (auto &blob : layer->blobs()) {
      int data_size = blob->count();
      float *blob_data = new float[data_size];
      blob->read_data(blob_data);
      shadow::Blob *layer_blob = net_param_.mutable_layer(l)->add_blobs();
      layer_blob->set_count(data_size);
      for (int i = 0; i < data_size; ++i) {
        layer_blob->add_data(blob_data[i]);
      }
      delete[] blob_data;
    }
  }
  IO::WriteProtoToBinaryFile(net_param_, proto_bin);
}

void Network::Forward(const float *data) {
  if (data == nullptr) Fatal("Must provide input data!");
  if (layers_.size() == 0) return;
  if (!layers_[0]->type().compare("Data")) {
    layers_[0]->bottom(0)->set_data(data);
  } else {
    Fatal("The first layer must be Data layer!");
  }
  for (int l = 0; l < layers_.size(); ++l) layers_[l]->Forward();
}

void Network::Release() {
  for (auto &layer : layers_) {
    delete layer;
    layer = nullptr;
  }
  for (auto &blob : blobs_) {
    delete blob;
    blob = nullptr;
  }

  net_param_.Clear();
  in_shape_.clear();
  layers_.clear();
  blobs_.clear();

  DInfo("Release Network!");
}

const Layer *Network::GetLayerByName(const std::string &layer_name) {
  for (const auto &layer : layers_) {
    if (!layer_name.compare(layer->name())) return layer;
  }
  return nullptr;
}

const Blob<float> *Network::GetBlobByName(const std::string &blob_name) {
  return get_blob_by_name(blobs_, blob_name);
}

void Network::LoadProtoBin(const std::string &proto_bin,
                           shadow::NetParameter *net_param) {
  if (!IO::ReadProtoFromBinaryFile(proto_bin, net_param)) {
    Fatal("Error when loading proto binary file: " + proto_bin);
  }
}

void Network::LoadProtoStrOrText(const std::string &proto_str_or_text,
                                 shadow::NetParameter *net_param) {
  bool success;
  Path path(proto_str_or_text);
  if (path.is_file()) {
    success = IO::ReadProtoFromTextFile(proto_str_or_text, net_param);
  } else {
    success = IO::ReadProtoFromText(proto_str_or_text, net_param);
  }
  if (!proto_str_or_text.compare("") || !success) {
    Fatal("Error when loading proto: " + proto_str_or_text);
  }
}

void Network::Reshape(int batch) {
  in_shape_.clear();
  for (const auto &dim : net_param_.input_shape().dim()) {
    in_shape_.push_back(dim);
  }
  if (in_shape_.size() != 4) Fatal("input_shape dimension mismatch!");
  if (batch > 0) in_shape_[0] = batch;

  Blob<float> *in_blob = new Blob<float>("in_blob");
  in_blob->reshape(in_shape_);
  blobs_.clear();
  blobs_.push_back(in_blob);

  layers_.clear();
  for (const auto &layer_param : net_param_.layer()) {
    Layer *layer = LayerFactory(layer_param);
    layer->Setup(&blobs_);
    layer->Reshape();
    layers_.push_back(layer);
  }
}

void Network::CopyWeights(const std::vector<const float *> &weights) {
  int count = 0;
  for (auto &layer : layers_) {
    if (layer->num_blobs() > 0) {
      CHECK_LT(count, weights.size());
      const float *weights_data = weights[count++];
      for (int n = 0; n < layer->num_blobs(); ++n) {
        layer->set_blob(n, weights_data);
        weights_data += layer->blob(n)->count();
      }
    }
  }
}

void Network::CopyWeights(const float *weights_data) {
  for (auto &layer : layers_) {
    for (int n = 0; n < layer->num_blobs(); ++n) {
      layer->set_blob(n, weights_data);
      weights_data += layer->blob(n)->count();
    }
  }
}

Layer *Network::LayerFactory(const shadow::LayerParameter &layer_param) {
  Layer *layer = nullptr;
  const auto &layer_type = layer_param.type();
  if (!layer_type.compare("Activate")) {
    layer = new ActivateLayer(layer_param);
  } else if (!layer_type.compare("Concat")) {
    layer = new ConcatLayer(layer_param);
  } else if (!layer_type.compare("Connected")) {
    layer = new ConnectedLayer(layer_param);
  } else if (!layer_type.compare("Convolution")) {
    layer = new ConvolutionLayer(layer_param);
  } else if (!layer_type.compare("Data")) {
    layer = new DataLayer(layer_param);
  } else if (!layer_type.compare("Flatten")) {
    layer = new FlattenLayer(layer_param);
  } else if (!layer_type.compare("Normalize")) {
    layer = new NormalizeLayer(layer_param);
  } else if (!layer_type.compare("Permute")) {
    layer = new PermuteLayer(layer_param);
  } else if (!layer_type.compare("Pooling")) {
    layer = new PoolingLayer(layer_param);
  } else if (!layer_type.compare("PriorBox")) {
    layer = new PriorBoxLayer(layer_param);
  } else if (!layer_type.compare("Reshape")) {
    layer = new ReshapeLayer(layer_param);
  } else if (!layer_type.compare("Softmax")) {
    layer = new SoftmaxLayer(layer_param);
  } else {
    Fatal("Error when making layer: " + layer_param.name() + ", layer type: " +
          layer_type + " is not recognized!");
  }
  return layer;
}
