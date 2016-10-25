#include "shadow/network.hpp"
#include "shadow/util/io.hpp"

#include "shadow/layers/activate_layer.hpp"
#include "shadow/layers/concat_layer.hpp"
#include "shadow/layers/connected_layer.hpp"
#include "shadow/layers/conv_layer.hpp"
#include "shadow/layers/data_layer.hpp"
#include "shadow/layers/dropout_layer.hpp"
#include "shadow/layers/flatten_layer.hpp"
#include "shadow/layers/normalize_layer.hpp"
#include "shadow/layers/permute_layer.hpp"
#include "shadow/layers/pooling_layer.hpp"

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

void Network::LoadModel(const std::string &proto_text,
                        const std::string &weights_file, int batch) {
  LoadProtoStrOrText(proto_text, &net_param_);
  Reshape(batch);
  CopyWeights(weights_file);
}

void Network::SaveModel(const std::string &proto_bin) {
  for (int l = 0; l < layers_.size(); ++l) {
    net_param_.mutable_layer(l)->clear_blobs();
    const Layer *layer = layers_[l];
    for (int n = 0; n < layer->num_blobs(); ++n) {
      int data_size = layer->blob(n)->count();
      float *blob_data = new float[data_size];
      layer->blob(n)->read_data(blob_data);
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

void Network::Forward(float *data) {
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
  for (int l = 0; l < layers_.size(); ++l) {
    layers_[l]->Release();
    layers_[l] = nullptr;
  }
  for (int n = 0; n < blobs_.size(); ++n) {
    blobs_[n]->clear();
    blobs_[n] = nullptr;
  }

  layers_.clear();
  blobs_.clear();

  DInfo("Release Network!");
}

const Layer *Network::GetLayerByName(const std::string &layer_name) {
  for (int l = 0; l < layers_.size(); ++l) {
    if (!layer_name.compare(layers_[l]->name()))
      return (const Layer *)layers_[l];
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
  for (int i = 0; i < net_param_.input_shape().dim_size(); ++i) {
    in_shape_.push_back(net_param_.input_shape().dim(i));
  }
  if (in_shape_.size() != 4) Fatal("input_shape dimension mismatch!");
  if (batch > 0) in_shape_[0] = batch;

  Blob<float> *in_blob = new Blob<float>("in_blob");
  in_blob->reshape(in_shape_);
  blobs_.push_back(in_blob);

  for (int l = 0; l < net_param_.layer_size(); ++l) {
    layers_.push_back(LayerFactory(net_param_.layer(l), &blobs_));
  }
}

void Network::CopyWeights(const float *weights_data) {
  for (int l = 0; l < layers_.size(); ++l) {
    Layer *layer = layers_[l];
    std::string layer_type = layer->type();
    if (!layer_type.compare("Connected") | !layer_type.compare("Convolution")) {
      for (int n = 0; n < layer->num_blobs(); ++n) {
        layer->set_blob(n, weights_data);
        weights_data += layer->blob(n)->count();
      }
    }
  }
}

void Network::CopyWeights(const std::string &weights_file) {
  DInfo("Load model from " + weights_file + " ... ");

  std::ifstream file(weights_file, std::ios::binary);
  if (!file.is_open()) Fatal("Load weight file error!");

  file.seekg(sizeof(char) * 16, std::ios::beg);

  for (int l = 0; l < layers_.size(); ++l) {
    Layer *layer = layers_[l];
    std::string layer_type = layer->type();
    if (!layer_type.compare("Connected") | !layer_type.compare("Convolution")) {
      for (int n = layer->num_blobs() - 1; n >= 0; --n) {
        int count = layer->blob(n)->count();
        float *weights = new float[count];
        file.read(reinterpret_cast<char *>(weights), count * sizeof(float));
        layer->set_blob(n, weights);
        delete[] weights;
      }
    }
  }

  file.close();
}

Layer *Network::LayerFactory(const shadow::LayerParameter &layer_param,
                             VecBlob *blobs) {
  Layer *layer = nullptr;
  if (!layer_param.type().compare("Activate")) {
    layer = new ActivateLayer(layer_param);
  } else if (!layer_param.type().compare("Concat")) {
    layer = new ConcatLayer(layer_param);
  } else if (!layer_param.type().compare("Connected")) {
    layer = new ConnectedLayer(layer_param);
  } else if (!layer_param.type().compare("Convolution")) {
    layer = new ConvLayer(layer_param);
  } else if (!layer_param.type().compare("Data")) {
    layer = new DataLayer(layer_param);
  } else if (!layer_param.type().compare("Dropout")) {
    layer = new DropoutLayer(layer_param);
  } else if (!layer_param.type().compare("Flatten")) {
    layer = new FlattenLayer(layer_param);
  } else if (!layer_param.type().compare("Normalize")) {
    layer = new NormalizeLayer(layer_param);
  } else if (!layer_param.type().compare("Permute")) {
    layer = new PermuteLayer(layer_param);
  } else if (!layer_param.type().compare("Pooling")) {
    layer = new PoolingLayer(layer_param);
  } else {
    Fatal("Layer type is not recognized!");
  }
  if (layer != nullptr) {
    layer->Setup(blobs);
    layer->Reshape();
  } else {
    Fatal("Error when making layer: " + layer_param.name());
  }
  return layer;
}
