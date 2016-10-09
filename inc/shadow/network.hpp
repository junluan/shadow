#ifndef SHADOW_NETWORK_HPP
#define SHADOW_NETWORK_HPP

#include "shadow/layers/layer.hpp"

class Network {
 public:
  void LoadModel(const std::string cfg_file, const std::string weight_file,
                 int batch = 1);
  void LoadModel(const std::string cfg_str, const float *weight_data,
                 int batch = 1);

  void Forward(float *in_data = nullptr);
  const Layer *GetLayerByName(const std::string layer_name);
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

#endif  // SHADOW_NETWORK_HPP
