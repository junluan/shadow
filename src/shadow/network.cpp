#include "shadow/network.hpp"
#include "shadow/util/parser.hpp"

void Network::LoadModel(std::string cfg_file, std::string weight_file,
                        int batch) {
  Parser parser;
  parser.ParseNetworkProto(this, cfg_file, batch);
  parser.LoadWeights(this, weight_file);
}

void Network::Forward(float *in_data) { ForwardNetwork(in_data); }

const Layer *Network::GetLayerByName(std::string layer_name) {
  for (int i = 0; i < num_layers_; ++i) {
    if (!layer_name.compare(layers_[i]->layer_param_.name())) {
      return (const Layer *)layers_[i];
    }
  }
  return nullptr;
}

void Network::ReleaseNetwork() {
  for (int i = 0; i < num_layers_; ++i)
    layers_[i]->ReleaseLayer();
#ifdef VERBOSE
  std::cout << "Release Network!" << std::endl;
#endif
}

void Network::ForwardNetwork(float *in_data) {
  for (int i = 0; i < num_layers_; ++i) {
    if (layers_[i]->layer_param_.type() == shadow::LayerType::Data)
      layers_[i]->ForwardLayer(in_data);
    else
      layers_[i]->ForwardLayer();
    if (i < num_layers_ - 1) {
      layers_[i + 1]->in_blob_ = layers_[i]->out_blob_;
    }
  }
}
