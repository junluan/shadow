#include "parser.hpp"
#include "kernel.hpp"
#include "util.hpp"

#include <fstream>

bool ParaFindBool(const Json::Value &params, std::string key, bool def) {
  return params.isMember(key) ? params[key].asBool() : def;
}

float ParaFindFloat(const Json::Value &params, std::string key, float def) {
  return params.isMember(key) ? params[key].asFloat() : def;
}

int ParaFindInt(const Json::Value &params, std::string key, int def) {
  return params.isMember(key) ? params[key].asInt() : def;
}

std::string ParaFindString(const Json::Value &params, std::string key,
                           std::string def) {
  return params.isMember(key) ? params[key].asString() : def;
}

float *ParaFindArrayFloat(const Json::Value &params, std::string key) {
  float *arr = nullptr;
  if (params.isMember(key)) {
    Json::Value array = params[key];
    arr = new float[array.size()];
    for (int i = 0; i < array.size(); ++i) {
      arr[i] = array[i].asFloat();
    }
  }
  return arr;
}

int *ParaFindArrayInt(const Json::Value &params, std::string key) {
  int *arr = nullptr;
  if (params.isMember(key)) {
    Json::Value array = params[key];
    arr = new int[array.size()];
    for (int i = 0; i < array.size(); ++i) {
      arr[i] = array[i].asInt();
    }
  }
  return arr;
}

void Parser::LoadWeights(Network *net, std::string weight_file) {
  LoadWeightsUpto(net, weight_file, net->num_layers_);
}

void Parser::ParseNetworkCfg(Network *net, std::string cfg_file, int batch) {
  std::ifstream file(cfg_file);
  Json::Reader reader;
  Json::Value root;
  if (!reader.parse(file, root, false))
    Fatal("Parse configure file error");
  Json::Value sections = root["network"];

  net->num_layers_ = sections.size() - 1;
  net->layers_.reserve(net->num_layers_);
  ParseNet(net, sections[0]);
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
    Json::Value section = sections[i + 1];
    std::string layer_type = section["type"].asString();
    Layer *l = nullptr;
    if (!layer_type.compare("Data")) {
      l = ParseData(section, params);
    } else if (!layer_type.compare("Convolution")) {
      l = ParseConvolutional(section, params);
    } else if (!layer_type.compare("Pooling")) {
      l = ParsePooling(section, params);
    } else if (!layer_type.compare("Connected")) {
      l = ParseConnected(section, params);
    } else if (!layer_type.compare("Dropout")) {
      l = ParseDropout(section, params);
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

void Parser::ParseNet(Network *net, Json::Value section) {
  Json::Value layer_params = section["parameters"];

  net->in_c_ = ParaFindInt(layer_params, "channels", 0);
  net->in_h_ = ParaFindInt(layer_params, "height", 0);
  net->in_w_ = ParaFindInt(layer_params, "width", 0);
  net->class_num_ = ParaFindInt(layer_params, "class_num", 1);
  net->grid_size_ = ParaFindInt(layer_params, "grid_size", 7);
  net->sqrt_box_ = ParaFindInt(layer_params, "sqrt_box", 1);
  net->box_num_ = ParaFindInt(layer_params, "box_num", 2);

  net->in_num_ = net->in_c_ * net->in_h_ * net->in_w_;
  if (!net->in_num_ && !(net->in_c_ && net->in_h_ && net->in_w_))
    Fatal("No input parameters supplied");
}

DataLayer *Parser::ParseData(Json::Value section, SizeParams params) {
  if (!(params.in_c && params.in_h && params.in_w))
    Fatal("Channel, height and width must greater than zero.");

  Json::Value layer_params = section["parameters"];

  DataLayer *data_layer = new DataLayer(kData);
  data_layer->MakeDataLayer(params);
  data_layer->layer_name_ = section["name"].asString();
  data_layer->scale_ = ParaFindFloat(layer_params, "scale", 1);
  data_layer->mean_value_ = ParaFindFloat(layer_params, "mean_value", 0);

  return data_layer;
}

ConvLayer *Parser::ParseConvolutional(Json::Value section, SizeParams params) {
  if (!(params.in_c && params.in_h && params.in_w))
    Fatal("Channel, height and width must greater than zero.");

  Json::Value layer_params = section["parameters"];

  int out_num = ParaFindInt(layer_params, "num_output", 1);
  int ksize = ParaFindInt(layer_params, "kernel_size", 3);
  int stride = ParaFindInt(layer_params, "stride", 1);
  int pad = ParaFindInt(layer_params, "pad", 0);
  std::string activation = ParaFindString(layer_params, "activation", "leaky");

  ConvLayer *conv_layer = new ConvLayer(kConvolutional);
  conv_layer->MakeConvLayer(params, out_num, ksize, stride, pad, activation);
  conv_layer->layer_name_ = section["name"].asString();

  return conv_layer;
}

PoolingLayer *Parser::ParsePooling(Json::Value section, SizeParams params) {
  if (!(params.in_c && params.in_h && params.in_w))
    Fatal("Channel, height and width must greater than zero.");

  Json::Value layer_params = section["parameters"];

  std::string pool_type = ParaFindString(layer_params, "pool", "Max");
  int ksize = ParaFindInt(layer_params, "kernel_size", 2);
  int stride = ParaFindInt(layer_params, "stride", 2);

  PoolingLayer *pooling_layer = new PoolingLayer(kMaxpool);
  pooling_layer->MakePoolingLayer(params, ksize, stride, pool_type);
  pooling_layer->layer_name_ = section["name"].asString();

  return pooling_layer;
}

ConnectedLayer *Parser::ParseConnected(Json::Value section, SizeParams params) {
  if (!(params.in_num))
    Fatal("input dimension must greater than zero.");

  Json::Value layer_params = section["parameters"];

  int out_num = ParaFindInt(layer_params, "num_output", 1);
  std::string activation = ParaFindString(layer_params, "activation", "leaky");

  ConnectedLayer *conn_layer = new ConnectedLayer(kConnected);
  conn_layer->MakeConnectedLayer(params, out_num, activation);
  conn_layer->layer_name_ = section["name"].asString();

  return conn_layer;
}

DropoutLayer *Parser::ParseDropout(Json::Value section, SizeParams params) {
  if (!(params.in_num))
    Fatal("input dimension must greater than zero.");

  Json::Value layer_params = section["parameters"];

  float probability = ParaFindFloat(layer_params, "probability", 0.5);
  DropoutLayer *drop_layer = new DropoutLayer(kDropout);
  drop_layer->MakeDropoutLayer(params, probability);
  drop_layer->layer_name_ = section["name"].asString();

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
