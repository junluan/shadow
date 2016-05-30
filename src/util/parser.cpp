#include "parser.hpp"
#include "kernel.hpp"
#include "util.hpp"

#include "connected_layer.hpp"
#include "conv_layer.hpp"
#include "data_layer.hpp"
#include "dropout_layer.hpp"
#include "pooling_layer.hpp"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

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
  net->batch_ = batch;

  shadow::BlobShape shape;
  shape.add_dim(net->batch_);
  shape.add_dim(net->in_c_);
  shape.add_dim(net->in_h_);
  shape.add_dim(net->in_w_);

  for (int i = 0; i < net->num_layers_; ++i) {
#ifdef VERBOSE
    printf("%2d: ", i);
#endif
    shadow::LayerParameter layer_param = net->net_param_.layer(i);
    Layer *layer = LayerFactory(layer_param, &shape);
    net->layers_.push_back(layer);
  }

  net->out_num_ = net->GetNetworkOutputSize();
}

void Parser::ParseNet(Network *net) {
  net->num_layers_ = net->net_param_.layer_size();
  net->layers_.reserve(net->num_layers_);

  net->in_c_ = net->net_param_.input_shape().dim(1);
  net->in_h_ = net->net_param_.input_shape().dim(2);
  net->in_w_ = net->net_param_.input_shape().dim(3);
  net->class_num_ = net->net_param_.class_num();
  net->grid_size_ = net->net_param_.grid_size();
  net->sqrt_box_ = net->net_param_.sqrt_box();
  net->box_num_ = net->net_param_.box_num();

  net->in_num_ = net->in_c_ * net->in_h_ * net->in_w_;
  if (!net->in_num_ && !(net->in_c_ && net->in_h_ && net->in_w_))
    Fatal("No input parameters supplied!");
}

Layer *Parser::LayerFactory(shadow::LayerParameter layer_param,
                            shadow::BlobShape *shape) {
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
    layer->MakeLayer(shape);
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
    if (layer->layer_type_ == shadow::LayerType::Convolution) {
      ConvLayer *l = reinterpret_cast<ConvLayer *>(layer);
      int in_c = l->in_blob->shape().dim(1),
          out_c = l->out_blob->shape().dim(1);
      int num = out_c * in_c * l->kernel_size_ * l->kernel_size_;
      file.read(reinterpret_cast<char *>(l->biases_), sizeof(float) * out_c);
      file.read(reinterpret_cast<char *>(l->filters_), sizeof(float) * num);

#ifdef USE_CUDA
      CUDA::CUDAWriteBuffer(out_c, l->cuda_biases_, l->biases_);
      CUDA::CUDAWriteBuffer(num, l->cuda_filters_, l->filters_);
#endif

#ifdef USE_CL
      CL::CLWriteBuffer(l->out_c_, l->cl_biases_, l->biases_);
      CL::CLWriteBuffer(num, l->cl_filters_, l->filters_);
#endif
    }
    if (layer->layer_type_ == shadow::LayerType::Connected) {
      ConnectedLayer *l = reinterpret_cast<ConnectedLayer *>(layer);
      int in_num = l->in_blob->num(), out_num = l->out_blob->num();
      file.read(reinterpret_cast<char *>(l->biases_), sizeof(float) * out_num);
      file.read(reinterpret_cast<char *>(l->weights_),
                sizeof(float) * in_num * out_num);

#ifdef USE_CUDA
      CUDA::CUDAWriteBuffer(out_num, l->cuda_biases_, l->biases_);
      CUDA::CUDAWriteBuffer(in_num * out_num, l->cuda_weights_, l->weights_);
#endif

#ifdef USE_CL
      CL::CLWriteBuffer(l->out_num_, l->cl_biases_, l->biases_);
      CL::CLWriteBuffer(l->in_num_ * l->out_num_, l->cl_weights_, l->weights_);
#endif
    }
  }

  file.close();
}
