#ifndef SHADOW_CORE_NETWORK_IMPL_HPP
#define SHADOW_CORE_NETWORK_IMPL_HPP

#include "network.hpp"

#include "operator.hpp"
#include "workspace.hpp"

namespace Shadow {

class Network::NetworkImpl {
 public:
  NetworkImpl() = default;
  ~NetworkImpl() { Release(); }

  void Setup(int device_id = 0);

  void LoadModel(const std::string &proto_bin);
  void LoadModel(const shadow::NetParam &net_param);
  void LoadModel(const std::string &proto_str,
                 const std::vector<const void *> &weights);
  void LoadModel(const std::string &proto_str, const float *weights_data);

  void Forward(const std::map<std::string, float *> &data_map,
               const std::map<std::string, std::vector<int>> &shape_map = {});
  void Release();

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

  template <typename T>
  const std::vector<int> GetBlobShapeByName(const std::string &blob_name) {
    auto *blob = ws_.GetBlob<T>(blob_name);
    if (blob == nullptr) {
      LOG(FATAL) << "Unknown blob: " + blob_name;
    } else {
      return blob->shape();
    }
    return std::vector<int>();
  }

  const std::vector<std::string> in_blob() { return in_blob_; }
  const std::vector<std::string> out_blob() { return out_blob_; }

  bool has_argument(const std::string &name) const {
    return arg_helper_.HasArgument(name);
  }
  template <typename T>
  T get_single_argument(const std::string &name, const T &default_value) const {
    return arg_helper_.template GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  bool has_single_argument_of_type(const std::string &name) const {
    return arg_helper_.template HasSingleArgumentOfType<T>(name);
  }
  template <typename T>
  const std::vector<T> get_repeated_argument(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    return arg_helper_.template GetRepeatedArgument<T>(name, default_value);
  }

 private:
  void LoadProtoBin(const std::string &proto_bin, shadow::NetParam *net_param);
  void LoadProtoStrOrText(const std::string &proto_str_or_text,
                          shadow::NetParam *net_param);

  void Initial();

  void CopyWeights(const std::vector<const void *> &weights);
  void CopyWeights(const float *weights_data);

  shadow::NetParam net_param_;
  ArgumentHelper arg_helper_;

  std::vector<Operator *> ops_;
  Workspace ws_;

  std::vector<std::string> in_blob_, out_blob_;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_IMPL_HPP
