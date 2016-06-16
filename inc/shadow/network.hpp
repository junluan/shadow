#ifndef SHADOW_NETWORK_HPP
#define SHADOW_NETWORK_HPP

#include "shadow/layers/layer.hpp"

#include <string>
#include <vector>

class Network {
public:
  void LoadModel(std::string cfg_file, std::string weight_file, int batch = 1);

  void Forward(float *in_data = nullptr);
  const Layer *GetLayerByName(std::string layer_name);
  void Release();

  shadow::NetParameter net_param_;
  shadow::BlobShape in_shape_;

  int num_layers_;
  VecLayer layers_;
  VecBlob blobs_;

private:
  void PreFillData(float *in_data);
  void ForwardNetwork();
};

#endif // SHADOW_NETWORK_HPP
