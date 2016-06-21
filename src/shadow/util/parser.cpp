#include "shadow/util/parser.hpp"
#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

#include "shadow/layers/connected_layer.hpp"
#include "shadow/layers/conv_layer.hpp"
#include "shadow/layers/data_layer.hpp"
#include "shadow/layers/dropout_layer.hpp"
#include "shadow/layers/pooling_layer.hpp"

#include <google/protobuf/text_format.h>

using google::protobuf::TextFormat;

void Parser::ParseNetworkProtoTxt(Network *net, const std::string prototxt_file,
                                  int batch) {
  std::string proto_str = read_text_from_file(prototxt_file);
  bool success = TextFormat::ParseFromString(proto_str, &net->net_param_);

  if (!proto_str.compare("") || !success) Fatal("Parse configure file error");

  ParseNet(net);
  net->in_shape_.set_dim(0, batch);

  Blob *in_blob = new Blob("in_blob");
  in_blob->set_shape(net->in_shape_);
  in_blob->allocate_data(in_blob->count());
  net->blobs_.push_back(in_blob);

  for (int i = 0; i < net->num_layers_; ++i) {
#if defined(VERBOSE)
    std::cout << format_int(i, 2) << ": ";
#endif

    shadow::LayerParameter layer_param = net->net_param_.layer(i);
    Layer *layer = LayerFactory(layer_param, &net->blobs_);
    net->layers_.push_back(layer);
  }
}

void Parser::LoadWeights(Network *net, const std::string weight_file) {
  LoadWeightsUpto(net, weight_file, net->num_layers_);
}

void Parser::ParseNet(Network *net) {
  net->num_layers_ = net->net_param_.layer_size();
  net->layers_.reserve(net->num_layers_);
  net->blobs_.reserve(net->num_layers_);

  net->in_shape_ = net->net_param_.input_shape();
  if (!(net->in_shape_.dim(1) && net->in_shape_.dim(2) &&
        net->in_shape_.dim(3)))
    Fatal("No input parameters supplied!");
}

Layer *Parser::LayerFactory(const shadow::LayerParameter &layer_param,
                            VecBlob *blobs) {
  shadow::LayerType layer_type = layer_param.type();
  Layer *layer = nullptr;
  if (layer_type == shadow::LayerType::Data) {
    layer = new DataLayer(layer_param);
  } else if (layer_type == shadow::LayerType::Convolution) {
    layer = new ConvLayer(layer_param);
  } else if (layer_type == shadow::LayerType::Pooling) {
    layer = new PoolingLayer(layer_param);
  } else if (layer_type == shadow::LayerType::Connected) {
    layer = new ConnectedLayer(layer_param);
  } else if (layer_type == shadow::LayerType::Dropout) {
    layer = new DropoutLayer(layer_param);
  } else {
    Fatal("Type not recognized!");
  }
  if (layer != nullptr)
    layer->Setup(blobs);
  else
    Fatal("Make layer error!");
  return layer;
}

void Parser::LoadWeightsUpto(Network *net, const std::string weight_file,
                             int cut_off) {
#if defined(VERBOSE)
  std::cout << "Load model from " << weight_file << " ... " << std::endl;
#endif
  std::ifstream file(weight_file, std::ios::binary);
  if (!file.is_open()) Fatal("Load weight file error!");

  char garbage[16];
  file.read(garbage, sizeof(char) * 16);

  for (int i = 0; i < net->num_layers_ && i < cut_off; ++i) {
    Layer *layer = net->layers_[i];
    if (layer->layer_param_.type() == shadow::LayerType::Convolution) {
      ConvLayer *l = reinterpret_cast<ConvLayer *>(layer);
      int in_c = l->bottom_[0]->shape(1), out_c = l->top_[0]->shape(1);
      int num = out_c * in_c * l->kernel_size() * l->kernel_size();
      float *biases = new float[out_c], *filters = new float[num];
      file.read(reinterpret_cast<char *>(biases), sizeof(float) * out_c);
      file.read(reinterpret_cast<char *>(filters), sizeof(float) * num);
      l->set_biases(biases);
      l->set_filters(filters);
      delete[] biases;
      delete[] filters;
    }
    if (layer->layer_param_.type() == shadow::LayerType::Connected) {
      ConnectedLayer *l = reinterpret_cast<ConnectedLayer *>(layer);
      int out_num = l->top_[0]->num(), num = l->bottom_[0]->num() * out_num;
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
