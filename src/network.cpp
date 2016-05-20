#include "network.hpp"
#include "parser.hpp"

void Network::LoadModel(std::string cfg_file, std::string weight_file,
                        int batch) {
  Parser parser;
  parser.ParseNetworkCfg(this, cfg_file, batch);
  parser.LoadWeights(this, weight_file);
}

float *Network::PredictNetwork(float *in_data) {
#ifdef USE_CUDA
  CUDAForwardNetwork(in_data);
#else
#ifdef USE_CL
  CLForwardNetwork(in_data);
#else
  ForwardNetwork(in_data);
#endif
#endif
  return GetNetworkOutput();
}

void Network::ReleaseNetwork() {
  for (int i = 0; i < num_layers_; ++i)
    layers_[i]->ReleaseLayer();
#ifdef VERBOSE
  std::cout << "Release Network!" << std::endl;
#endif
}

int Network::GetNetworkOutputSize() {
  int i;
  for (i = num_layers_ - 1; i > 0; --i)
    if (layers_[i]->layer_type_ != kCost)
      break;
  return layers_[i]->out_num_;
}

float *Network::GetNetworkOutput() {
  int i;
  for (i = num_layers_ - 1; i > 0; --i)
    if (layers_[i]->layer_type_ != kCost)
      break;
  return layers_[i]->GetOutData();
}

void Network::ForwardNetwork(float *in_data) {
  for (int i = 0; i < num_layers_; ++i) {
    if (layers_[i]->layer_type_ == kData)
      layers_[i]->ForwardLayer(in_data);
    else
      layers_[i]->ForwardLayer();
    if (i < num_layers_ - 1) {
      layers_[i + 1]->in_data_ = layers_[i]->out_data_;
    }
  }
}

#ifdef USE_CUDA
void Network::CUDAForwardNetwork(float *in_data) {
  for (int i = 0; i < num_layers_; ++i) {
    if (layers_[i]->layer_type_ == kData)
      layers_[i]->CUDAForwardLayer(in_data);
    else
      layers_[i]->CUDAForwardLayer();
    if (i < num_layers_ - 1) {
      layers_[i + 1]->cuda_in_data_ = layers_[i]->cuda_out_data_;
    }
  }
}
#endif

#ifdef USE_CL
void Network::CLForwardNetwork(float *in_data) {
  for (int i = 0; i < num_layers_; ++i) {
    if (layers_[i]->layer_type_ == kData)
      layers_[i]->CLForwardLayer(in_data);
    else
      layers_[i]->CLForwardLayer();
    if (i < num_layers_ - 1) {
      layers_[i + 1]->cl_in_data_ = layers_[i]->cl_out_data_;
    }
  }
}
#endif
