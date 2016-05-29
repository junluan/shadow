#include "parser.hpp"
#include "kernel.hpp"
#include "util.hpp"

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

  SizeParams params;
  params.batch = net->batch_;
  params.in_c = net->in_c_;
  params.in_h = net->in_h_;
  params.in_w = net->in_w_;
  params.in_num = net->in_num_;

  for (int i = 0; i < net->num_layers_; ++i) {
#ifdef VERBOSE
    printf("%2d: ", i);
#endif
    shadow::LayerParameter layer_param = net->net_param_.layer(i);
    std::string layer_type = layer_param.type();
    Layer *l = nullptr;
    if (!layer_type.compare("Data")) {
      l = ParseData(layer_param, params);
    } else if (!layer_type.compare("Convolution")) {
      l = ParseConvolution(layer_param, params);
    } else if (!layer_type.compare("Pooling")) {
      l = ParsePooling(layer_param, params);
    } else if (!layer_type.compare("Connected")) {
      l = ParseConnected(layer_param, params);
    } else if (!layer_type.compare("Dropout")) {
      l = ParseDropout(layer_param, params);
      l->out_data_ = net->layers_[i - 1]->out_data_;
    } else {
      Fatal("Type not recognized");
    }
    net->layers_.push_back(l);
    if (i != net->num_layers_ - 1 && l != nullptr) {
      params.in_c = l->out_c_;
      params.in_h = l->out_h_;
      params.in_w = l->out_w_;
      params.in_num = l->out_num_;
    }
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
    Fatal("No input parameters supplied");
}

Layer *Parser::ParseData(shadow::LayerParameter layer_param,
                         SizeParams params) {
  if (!(params.in_c && params.in_h && params.in_w))
    Fatal("Channel, height and width must greater than zero.");

  shadow::DataParameter data_param = layer_param.data_param();

  DataLayer *data_layer = new DataLayer(kData);
  data_layer->MakeDataLayer(params);
  data_layer->layer_name_ = layer_param.name();
  data_layer->scale_ = data_param.scale();
  data_layer->mean_value_ = data_param.mean_value();

  return data_layer;
}

Layer *Parser::ParseConvolution(shadow::LayerParameter layer_param,
                                SizeParams params) {
  if (!(params.in_c && params.in_h && params.in_w))
    Fatal("Channel, height and width must greater than zero.");

  shadow::ConvolutionParameter conv_param = layer_param.convolution_param();

  int out_num = conv_param.num_output();
  int ksize = conv_param.kernel_size();
  int stride = conv_param.stride();
  int pad = conv_param.pad();
  std::string activation = conv_param.activation();

  ConvLayer *conv_layer = new ConvLayer(kConvolutional);
  conv_layer->MakeConvLayer(params, out_num, ksize, stride, pad, activation);
  conv_layer->layer_name_ = layer_param.name();

  return conv_layer;
}

Layer *Parser::ParsePooling(shadow::LayerParameter layer_param,
                            SizeParams params) {
  if (!(params.in_c && params.in_h && params.in_w))
    Fatal("Channel, height and width must greater than zero.");

  shadow::PoolingParameter pooling_param = layer_param.pooling_param();

  std::string pool_type = pooling_param.pool();
  int ksize = pooling_param.kernel_size();
  int stride = pooling_param.stride();

  PoolingLayer *pooling_layer = new PoolingLayer(kMaxpool);
  pooling_layer->MakePoolingLayer(params, ksize, stride, pool_type);
  pooling_layer->layer_name_ = layer_param.name();

  return pooling_layer;
}

Layer *Parser::ParseConnected(shadow::LayerParameter layer_param,
                              SizeParams params) {
  if (!(params.in_num))
    Fatal("input dimension must greater than zero.");

  shadow::ConnectedParameter conn_param = layer_param.connected_param();

  int out_num = conn_param.num_output();
  std::string activation = conn_param.activation();

  ConnectedLayer *conn_layer = new ConnectedLayer(kConnected);
  conn_layer->MakeConnectedLayer(params, out_num, activation);
  conn_layer->layer_name_ = layer_param.name();

  return conn_layer;
}

Layer *Parser::ParseDropout(shadow::LayerParameter layer_param,
                            SizeParams params) {
  if (!(params.in_num))
    Fatal("input dimension must greater than zero.");

  shadow::DropoutParameter dropout_param = layer_param.dropout_param();

  float probability = dropout_param.probability();
  DropoutLayer *drop_layer = new DropoutLayer(kDropout);
  drop_layer->MakeDropoutLayer(params, probability);
  drop_layer->layer_name_ = layer_param.name();

  return drop_layer;
}

void Parser::LoadWeightsUpto(Network *net, std::string weightfile, int cutoff) {
#ifdef VERBOSE
  std::cout << "Load model from " << weightfile << " ... " << std::endl;
#endif
  std::ifstream file(weightfile, std::ios::binary);
  if (!file.is_open())
    Fatal("Load weight file error!");

  char garbage[16];
  file.read(garbage, sizeof(char) * 16);

  for (int i = 0; i < net->num_layers_ && i < cutoff; ++i) {
    Layer *layer = net->layers_[i];
    if (layer->layer_type_ == kConvolutional) {
      ConvLayer *l = reinterpret_cast<ConvLayer *>(layer);
      int num = l->out_c_ * l->in_c_ * l->ksize_ * l->ksize_;
      file.read((char *)l->biases_, sizeof(float) * l->out_c_);
      file.read((char *)l->filters_, sizeof(float) * num);

#ifdef USE_CUDA
      CUDA::CUDAWriteBuffer(l->out_c_, l->cuda_biases_, l->biases_);
      CUDA::CUDAWriteBuffer(num, l->cuda_filters_, l->filters_);
#endif

#ifdef USE_CL
      CL::CLWriteBuffer(l->out_c_, l->cl_biases_, l->biases_);
      CL::CLWriteBuffer(num, l->cl_filters_, l->filters_);
#endif
    }
    if (layer->layer_type_ == kConnected) {
      ConnectedLayer *l = reinterpret_cast<ConnectedLayer *>(layer);
      file.read((char *)l->biases_, sizeof(float) * l->out_num_);
      file.read((char *)l->weights_, sizeof(float) * l->out_num_ * l->in_num_);

#ifdef USE_CUDA
      CUDA::CUDAWriteBuffer(l->out_num_, l->cuda_biases_, l->biases_);
      CUDA::CUDAWriteBuffer(l->in_num_ * l->out_num_, l->cuda_weights_,
                            l->weights_);
#endif

#ifdef USE_CL
      CL::CLWriteBuffer(l->out_num_, l->cl_biases_, l->biases_);
      CL::CLWriteBuffer(l->in_num_ * l->out_num_, l->cl_weights_, l->weights_);
#endif
    }
  }

  file.close();
}
