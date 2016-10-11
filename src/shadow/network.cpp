#include "shadow/network.hpp"

#include "shadow/layers/concat_layer.hpp"
#include "shadow/layers/connected_layer.hpp"
#include "shadow/layers/conv_layer.hpp"
#include "shadow/layers/data_layer.hpp"
#include "shadow/layers/dropout_layer.hpp"
#include "shadow/layers/flatten_layer.hpp"
#include "shadow/layers/permute_layer.hpp"
#include "shadow/layers/pooling_layer.hpp"

#include <google/protobuf/text_format.h>

using google::protobuf::TextFormat;

void Network::LoadModel(const std::string &proto_file,
                        const std::string &weight_file, int batch) {
  LoadProtoStr(Util::read_text_from_file(proto_file), batch);
  LoadWeights(weight_file);
}

void Network::LoadModel(const std::string &proto_str, const float *weight_data,
                        int batch) {
  LoadProtoStr(proto_str, batch);
  LoadWeights(weight_data);
}

void Network::Forward(float *in_data) {
  if (in_data != nullptr) PreFillData(in_data);
  for (int i = 0; i < layers_.size(); ++i) layers_[i]->Forward();
}

void Network::Release() {
  for (int i = 0; i < layers_.size(); ++i) layers_[i]->Release();
  for (int i = 0; i < blobs_.size(); ++i) blobs_[i]->clear();

  DInfo("Release Network!");
}

const Layer *Network::GetLayerByName(const std::string &layer_name) {
  for (int i = 0; i < layers_.size(); ++i) {
    if (!layer_name.compare(layers_[i]->name()))
      return (const Layer *)layers_[i];
  }
  return nullptr;
}

const Blob<float> *Network::GetBlobByName(const std::string &blob_name) {
  return get_blob_by_name(blobs_, blob_name);
}

void Network::LoadProtoStr(const std::string &proto_str, int batch) {
  bool success = TextFormat::ParseFromString(proto_str, &net_param_);

  if (!proto_str.compare("") || !success) Fatal("Parse configure file error!");

  Reshape(batch);

  Blob<float> *in_blob = new Blob<float>("in_blob");

  in_blob->set_shape(in_shape_);
  in_blob->allocate_data(in_blob->count());
  blobs_.push_back(in_blob);

  for (int i = 0; i < net_param_.layer_size(); ++i) {
    layers_.push_back(LayerFactory(net_param_.layer(i), &blobs_));
  }
}

void Network::LoadWeights(const std::string &weight_file) {
  DInfo("Load model from " + weight_file + " ... ");

  std::ifstream file(weight_file, std::ios::binary);
  if (!file.is_open()) Fatal("Load weight file error!");

  file.seekg(sizeof(char) * 16, std::ios::beg);

  for (int i = 0; i < layers_.size(); ++i) {
    Layer *layer = layers_[i];
    if (layer->type() == shadow::LayerType::Convolution) {
      ConvLayer *l = reinterpret_cast<ConvLayer *>(layer);
      int in_c = l->bottom(0)->shape(1), out_c = l->top(0)->shape(1);
      int num = out_c * in_c * l->kernel_size() * l->kernel_size();
      float *biases = new float[out_c], *filters = new float[num];
      file.read(reinterpret_cast<char *>(biases), sizeof(float) * out_c);
      file.read(reinterpret_cast<char *>(filters), sizeof(float) * num);
      l->set_biases(biases);
      l->set_filters(filters);
      delete[] biases;
      delete[] filters;
    }
    if (layer->type() == shadow::LayerType::Connected) {
      ConnectedLayer *l = reinterpret_cast<ConnectedLayer *>(layer);
      int out_num = l->top(0)->num(), num = l->bottom(0)->num() * out_num;
      float *biases = new float[out_num], *weights = new float[num];
      file.read(reinterpret_cast<char *>(biases), sizeof(float) * out_num);
      file.read(reinterpret_cast<char *>(weights), sizeof(float) * num);
      l->set_biases(biases);
      l->set_weights(weights);
      delete[] biases;
      delete[] weights;
    }
  }

  file.close();
}

void Network::LoadWeights(const float *weight_data) {
  const float *index = weight_data;
  for (int i = 0; i < layers_.size(); ++i) {
    Layer *layer = layers_[i];
    if (layer->type() == shadow::LayerType::Convolution) {
      ConvLayer *l = reinterpret_cast<ConvLayer *>(layer);
      int in_c = l->bottom(0)->shape(1), out_c = l->top(0)->shape(1);
      int num = out_c * in_c * l->kernel_size() * l->kernel_size();
      l->set_biases(index);
      index += out_c;
      l->set_filters(index);
      index += num;
    }
    if (layer->type() == shadow::LayerType::Connected) {
      ConnectedLayer *l = reinterpret_cast<ConnectedLayer *>(layer);
      int out_num = l->top(0)->num(), num = l->bottom(0)->num() * out_num;
      l->set_biases(index);
      index += out_num;
      l->set_weights(index);
      index += num;
    }
  }
}

void Network::Reshape(int batch) {
  for (int i = 0; i < net_param_.input_shape().dim_size(); ++i) {
    in_shape_.push_back(net_param_.input_shape().dim(i));
  }
  if (in_shape_.size() != 4) Fatal("input_shape dimension mismatch!");
  if (batch != 0) in_shape_[0] = batch;
}

Layer *Network::LayerFactory(const shadow::LayerParameter &layer_param,
                             VecBlob *blobs) {
  Layer *layer = nullptr;
  if (layer_param.type() == shadow::LayerType::Data) {
    layer = new DataLayer(layer_param);
  } else if (layer_param.type() == shadow::LayerType::Convolution) {
    layer = new ConvLayer(layer_param);
  } else if (layer_param.type() == shadow::LayerType::Pooling) {
    layer = new PoolingLayer(layer_param);
  } else if (layer_param.type() == shadow::LayerType::Connected) {
    layer = new ConnectedLayer(layer_param);
  } else if (layer_param.type() == shadow::LayerType::Dropout) {
    layer = new DropoutLayer(layer_param);
  } else if (layer_param.type() == shadow::LayerType::Concat) {
    layer = new ConcatLayer(layer_param);
  } else if (layer_param.type() == shadow::LayerType::Permute) {
    layer = new PermuteLayer(layer_param);
  } else if (layer_param.type() == shadow::LayerType::Flatten) {
    layer = new FlattenLayer(layer_param);
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

void Network::PreFillData(float *in_data) {
  for (int i = 0; i < layers_.size(); ++i) {
    if (layers_[i]->type() == shadow::LayerType::Data) {
      layers_[i]->bottom(0)->set_data(in_data);
      break;
    }
  }
}
