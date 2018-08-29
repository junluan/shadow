#ifndef SHADOW_CORE_NETWORK_HPP
#define SHADOW_CORE_NETWORK_HPP

#include "operator.hpp"
#include "workspace.hpp"

namespace Shadow {

class Network {
 public:
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
  const Blob<T> *GetBlobByName(const std::string &blob_name) const {
    return ws_.GetBlob<T>(blob_name);
  }
  template <typename T>
  Blob<T> *GetBlobByName(const std::string &blob_name) {
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

  const std::vector<std::string> in_blob() {
    VecString in_blobs;
    for (const auto &blob : net_param_.op(0).top()) {
      in_blobs.push_back(blob);
    }
    return in_blobs;
  }
  const std::vector<std::string> out_blob() {
    CHECK(has_argument("out_blob")) << "Network must have out_blob argument";
    return get_repeated_argument<std::string>("out_blob");
  }

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
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_HPP
