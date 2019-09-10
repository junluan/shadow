#ifndef SHADOW_CORE_NETWORK_IMPL_HPP
#define SHADOW_CORE_NETWORK_IMPL_HPP

#include "network.hpp"

#include "backend.hpp"
#include "operator.hpp"
#include "workspace.hpp"

namespace Shadow {

class Network::NetworkImpl {
 public:
  void Setup(int device_id = 0);

  void LoadModel(const shadow::NetParam &net_param);
  void LoadModel(const void *proto_data, int proto_size);
  void LoadModel(const std::string &proto_bin);
  void LoadModel(const std::string &proto_str,
                 const std::vector<const void *> &weights);
  void LoadModel(const std::string &proto_str, const void *weights_data);

  void LoadXModel(const shadow::NetParam &net_param,
                  const ArgumentHelper &arguments);

  void Forward(const std::map<std::string, float *> &data_map,
               const std::map<std::string, std::vector<int>> &shape_map);

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
  std::vector<int> GetBlobShapeByName(const std::string &blob_name) const {
    auto *blob = ws_.GetBlob<T>(blob_name);
    if (blob == nullptr) {
      LOG(FATAL) << "Unknown blob: " + blob_name;
    } else {
      return blob->shape();
    }
    return std::vector<int>();
  }

  const std::vector<std::string> &in_blob() const { return in_blob_; }
  const std::vector<std::string> &out_blob() const { return out_blob_; }

  bool has_argument(const std::string &name) const {
    return arg_helper_.HasArgument(name);
  }
  template <typename T>
  T get_single_argument(const std::string &name, const T &default_value) const {
    return arg_helper_.template GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  std::vector<T> get_repeated_argument(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    return arg_helper_.template GetRepeatedArgument<T>(name, default_value);
  }

 private:
  static void LoadProtoData(const void *proto_data, int proto_size,
                            shadow::NetParam *net_param);
  static void LoadProtoBin(const std::string &proto_bin,
                           shadow::NetParam *net_param);
  static void LoadProtoStrOrText(const std::string &proto_str_or_text,
                                 shadow::NetParam *net_param);

  void Initial();

  void CopyWeights(const std::vector<const void *> &weights);
  void CopyWeights(const void *weights_data);

  Workspace ws_;
  std::vector<std::shared_ptr<Operator>> ops_;

  ArgumentHelper arg_helper_;
  shadow::NetParam net_param_;

  std::vector<std::string> in_blob_, out_blob_;

  std::shared_ptr<Backend> backend_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_IMPL_HPP
