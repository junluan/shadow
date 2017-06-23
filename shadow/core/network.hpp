#ifndef SHADOW_CORE_NETWORK_HPP
#define SHADOW_CORE_NETWORK_HPP

#include "operator.hpp"

namespace Shadow {

class Network {
 public:
  void Setup(int device_id = 0);

  void LoadModel(const std::string &proto_bin, int batch = 0);
  void LoadModel(const std::string &proto_str,
                 const std::vector<const float *> &weights, int batch = 0);
  void LoadModel(const std::string &proto_str, const float *weights_data,
                 int batch = 0);

  void SaveModel(const std::string &proto_bin);

  void Forward(const float *data = nullptr);
  void Release();

  const Operator *GetOpByName(const std::string &op_name);
  const BlobF *GetBlobByName(const std::string &blob_name);
  const float *GetBlobDataByName(const std::string &blob_name);

  const VecInt &in_shape() { return in_shape_; }

 private:
  void LoadProtoBin(const std::string &proto_bin, shadow::NetParam *net_param);
  void LoadProtoStrOrText(const std::string &proto_str_or_text,
                          shadow::NetParam *net_param);

  void Reshape(int batch);

  void CopyWeights(const std::vector<const float *> &weights);
  void CopyWeights(const float *weights_data);

  shadow::NetParam net_param_;

  VecInt in_shape_;
  VecOp ops_;
  VecBlobF blobs_;
  std::map<std::string, VecFloat> blobs_data_;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_HPP
