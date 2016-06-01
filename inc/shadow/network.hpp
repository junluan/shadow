#ifndef SHADOW_NETWORK_HPP
#define SHADOW_NETWORK_HPP

#include "shadow/layers/layer.hpp"

#include <string>
#include <vector>

class Network {
public:
  void LoadModel(std::string cfg_file, std::string weight_file, int batch = 1);

  void Forward(float *in_data);
  const Layer *GetLayerByName(std::string layer_name);
  void ReleaseNetwork();

  shadow::NetParameter net_param_;
  shadow::BlobShape in_shape_;

  int num_layers_;
  std::vector<Layer *> layers_;

private:
  void ForwardNetwork(float *in_data);
};

#endif // SHADOW_NETWORK_HPP
