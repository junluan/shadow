#ifndef SHADOW_CORE_NETWORK_HPP
#define SHADOW_CORE_NETWORK_HPP

#include "operator.hpp"
#include "workspace.hpp"

namespace Shadow {

class Network {
 public:
  void Setup(int device_id = 0);

  void LoadModel(const std::string &proto_bin, const VecInt &in_shape = {});
  void LoadModel(const std::string &proto_str,
                 const std::vector<const void *> &weights,
                 const VecInt &in_shape = {});
  void LoadModel(const std::string &proto_str, const float *weights_data,
                 const VecInt &in_shape = {});

  void Reshape(const VecInt &in_shape);
  void Forward(const float *data = nullptr);
  void Release();

  const Operator *GetOpByName(const std::string &op_name) {
    for (const auto &op : ops_) {
      if (op_name == op->name()) return op;
    }
    return nullptr;
  }
  template <typename T>
  const Blob<T> *GetBlobByName(const std::string &blob_name) {
    return ws_.GetBlob<T>(blob_name);
  }
  template <typename T>
  const T *GetBlobDataByName(const std::string &blob_name) {
    auto *blob = ws_.GetBlob<T>(blob_name);
    if (blob == nullptr) {
      LOG(FATAL) << "Unknown blob: " + blob_name;
    } else {
      return blob->cpu_data();
    }
    return nullptr;
  }

  const VecInt &in_shape() { return in_shape_; }

 private:
  void LoadProtoBin(const std::string &proto_bin, shadow::NetParam *net_param);
  void LoadProtoStrOrText(const std::string &proto_str_or_text,
                          shadow::NetParam *net_param);

  void Initial(const VecInt &in_shape);

  void CopyWeights(const std::vector<const void *> &weights);
  void CopyWeights(const float *weights_data);

  shadow::NetParam net_param_;

  VecInt in_shape_;
  VecOp ops_;
  Workspace ws_;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_HPP
