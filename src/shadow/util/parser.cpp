#include "shadow/util/parser.hpp"
#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

#include "shadow/layers/connected_layer.hpp"
#include "shadow/layers/conv_layer.hpp"
#include "shadow/layers/data_layer.hpp"
#include "shadow/layers/dropout_layer.hpp"
#include "shadow/layers/pooling_layer.hpp"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#ifdef __unix__
#include <fcntl.h>
#else
#include <io.h>
#endif

using google::protobuf::io::FileInputStream;
using google::protobuf::TextFormat;

void Parser::LoadWeights(Network *net, std::string weight_file) {
  LoadWeightsUpto(net, weight_file, net->num_layers_);
}

void Parser::ParseNetworkProto(Network *net, std::string prototxt_file,
                               int batch) {
  int fd = open(prototxt_file.c_str(), O_RDONLY);
  if (fd == -1)
    Fatal("File not found: " + prototxt_file);
  FileInputStream *input = new FileInputStream(fd);
  bool success = TextFormat::Parse(input, &net->net_param_);
  delete input;
  close(fd);
  if (!success)
    Fatal("Parse configure file error");

  ParseNet(net);
  net->in_shape_.set_dim(0, batch);

  Blob<BType> *blob = new Blob<BType>();
  blob->add_shape(net->in_shape_.dim(0));
  blob->add_shape(net->in_shape_.dim(1));
  blob->add_shape(net->in_shape_.dim(2));
  blob->add_shape(net->in_shape_.dim(3));

  for (int i = 0; i < net->num_layers_; ++i) {
#ifdef VERBOSE
    printf("%2d: ", i);
#endif
    shadow::LayerParameter layer_param = net->net_param_.layer(i);
    Layer *layer = LayerFactory(layer_param, blob);
    net->layers_.push_back(layer);
  }
}

void Parser::ParseNet(Network *net) {
  net->num_layers_ = net->net_param_.layer_size();
  net->layers_.reserve(net->num_layers_);

  net->in_shape_ = net->net_param_.input_shape();
  if (!(net->in_shape_.dim(1) && net->in_shape_.dim(2) &&
        net->in_shape_.dim(3)))
    Fatal("No input parameters supplied!");
}

Layer *Parser::LayerFactory(shadow::LayerParameter layer_param,
                            Blob<BType> *blob) {
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
    layer->MakeLayer(blob);
  else
    Fatal("Make layer error!");
  return layer;
}

void Parser::LoadWeightsUpto(Network *net, std::string weight_file,
                             int cut_off) {
#ifdef VERBOSE
  std::cout << "Load model from " << weight_file << " ... " << std::endl;
#endif
  std::ifstream file(weight_file, std::ios::binary);
  if (!file.is_open())
    Fatal("Load weight file error!");

  char garbage[16];
  file.read(garbage, sizeof(char) * 16);

  for (int i = 0; i < net->num_layers_ && i < cut_off; ++i) {
    Layer *layer = net->layers_[i];
    if (layer->layer_param_.type() == shadow::LayerType::Convolution) {
      ConvLayer *l = reinterpret_cast<ConvLayer *>(layer);
      int in_c = l->in_blob_->shape(1), out_c = l->out_blob_->shape(1);
      int num = out_c * in_c * l->kernel_size_ * l->kernel_size_;
      float *biases = new float[out_c], *filters = new float[num];
      file.read(reinterpret_cast<char *>(biases), sizeof(float) * out_c);
      file.read(reinterpret_cast<char *>(filters), sizeof(float) * num);

      l->biases_->set_data(biases);
      l->filters_->set_data(filters);

      delete[] biases;
      delete[] filters;

      //#ifdef USE_CL
      //      CL::CLWriteBuffer(out_c, l->cl_biases_, l->biases_);
      //      CL::CLWriteBuffer(num, l->cl_filters_, l->filters_);
      //#endif
    }
    if (layer->layer_param_.type() == shadow::LayerType::Connected) {
      ConnectedLayer *l = reinterpret_cast<ConnectedLayer *>(layer);
      int out_num = l->out_blob_->num(), num = l->in_blob_->num() * out_num;
      float *biases = new float[out_num], *weights = new float[num];
      file.read(reinterpret_cast<char *>(biases), sizeof(float) * out_num);
      file.read(reinterpret_cast<char *>(weights), sizeof(float) * num);

      l->biases_->set_data(biases);
      l->weights_->set_data(weights);

      delete[] biases;
      delete[] weights;

      //#ifdef USE_CL
      //      CL::CLWriteBuffer(out_num, l->cl_biases_, l->biases_);
      //      CL::CLWriteBuffer(in_num * out_num, l->cl_weights_, l->weights_);
      //#endif
    }
  }

  file.close();
}
