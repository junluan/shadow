#include "shadow/network.hpp"
#include "shadow/util/parser.hpp"

void Network::LoadModel(const std::string cfg_file,
                        const std::string weight_file, int batch) {
  Parser parser;
  parser.ParseNetworkProtoTxt(this, cfg_file, batch);
  parser.LoadWeights(this, weight_file);
}

void Network::Forward(float *in_data) {
  if (in_data != nullptr) PreFillData(in_data);
  ForwardNetwork();
}

const Layer *Network::GetLayerByName(const std::string layer_name) {
  for (int i = 0; i < num_layers_; ++i) {
    if (!layer_name.compare(layers_[i]->layer_param_.name()))
      return (const Layer *)layers_[i];
  }
  return nullptr;
}

void Network::Release() {
  for (int i = 0; i < num_layers_; ++i) layers_[i]->Release();
  for (int i = 0; i < blobs_.size(); ++i) blobs_[i]->clear();

#if defined(VERBOSE)
  std::cout << "Release Network!" << std::endl;
#endif
}

void Network::PreFillData(float *in_data) {
  for (int i = 0; i < num_layers_; ++i) {
    if (layers_[i]->layer_param_.type() == shadow::LayerType::Data) {
      layers_[i]->bottom_[0]->set_data(in_data);
      break;
    }
  }
}

void Network::ForwardNetwork() {
  for (int i = 0; i < num_layers_; ++i) {
    layers_[i]->Forward();
  }
}
