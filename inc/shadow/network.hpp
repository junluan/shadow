#ifndef SHADOW_NETWORK_HPP
#define SHADOW_NETWORK_HPP

#include "shadow/layers/layer.hpp"

class Network {
 public:
  void LoadModel(const std::string &proto_bin, int batch = 0);
  void LoadModel(const std::string &proto_str, const float *weights_data,
                 int batch = 0);
  void LoadModel(const std::string &proto_text, const std::string &weights_file,
                 int batch = 0);

  void SaveModel(const std::string &proto_bin);

  void Forward(float *data = nullptr);
  void Release();

  const Layer *GetLayerByName(const std::string &layer_name);
  const Blob<float> *GetBlobByName(const std::string &blob_name);

  const VecInt in_shape() { return in_shape_; }

 private:
  void LoadProtoBin(const std::string &proto_bin,
                    shadow::NetParameter *net_param);
  void LoadProtoStrOrText(const std::string &proto_str_or_text,
                          shadow::NetParameter *net_param);

  void Reshape(int batch);

  void CopyWeights(const float *weights_data);
  void CopyWeights(const std::string &weights_file);

  Layer *LayerFactory(const shadow::LayerParameter &layer_param,
                      VecBlob *blobs);

  shadow::NetParameter net_param_;

  VecInt in_shape_;
  VecLayer layers_;
  VecBlob blobs_;
};

#endif  // SHADOW_NETWORK_HPP
