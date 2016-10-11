#ifndef SHADOW_NETWORK_HPP
#define SHADOW_NETWORK_HPP

#include "shadow/layers/layer.hpp"

class Network {
 public:
  void LoadModel(const std::string &proto_file, const std::string &weight_file,
                 int batch = 1);
  void LoadModel(const std::string &proto_str, const float *weight_data,
                 int batch = 1);

  void Forward(float *in_data = nullptr);
  void Release();

  const Layer *GetLayerByName(const std::string &layer_name);
  const Blob<float> *GetBlobByName(const std::string &blob_name);

  VecInt in_shape_;

 private:
  void LoadProtoStr(const std::string &proto_str, int batch);
  void LoadWeights(const std::string &weight_file);
  void LoadWeights(const float *weight_data);

  void Reshape(int batch = 0);
  Layer *LayerFactory(const shadow::LayerParameter &layer_param,
                      VecBlob *blobs);

  void PreFillData(float *in_data);

  shadow::NetParameter net_param_;

  VecLayer layers_;
  VecBlob blobs_;
};

#endif  // SHADOW_NETWORK_HPP
