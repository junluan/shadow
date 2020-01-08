#ifndef SHADOW_CORE_NETWORK_IMPL_HPP
#define SHADOW_CORE_NETWORK_IMPL_HPP

#include "backend.hpp"

namespace Shadow {

class NetworkImpl {
 public:
  void Setup(int device_id = 0) { ws_.CreateCtx(device_id); }

  void LoadXModel(const shadow::NetParam &net_param,
                  const ArgumentHelper &arguments) {
    backend_.reset(CreateBackend(arguments, &ws_));
    backend_->LoadModel(net_param);
  }

  void Forward(const std::map<std::string, void *> &data_map,
               const std::map<std::string, std::vector<int>> &shape_map) {
    ws_.Ctx()->SwitchDevice();

    CHECK_NOTNULL(backend_);
    backend_->Forward(data_map, shape_map);

    ws_.Ctx()->Synchronize();
  }

  template <typename T>
  const T *GetBlobDataByName(const std::string &blob_name,
                             const std::string &locate) {
    auto *blob = ws_.GetBlob<T>(blob_name);
    CHECK_NOTNULL(blob) << "Unknown blob: " + blob_name;
    if (locate == "host") {
      return blob->cpu_data();
    } else {
      return blob->data();
    }
  }

  template <typename T>
  std::vector<int> GetBlobShapeByName(const std::string &blob_name) const {
    auto *blob = ws_.GetBlob<T>(blob_name);
    CHECK_NOTNULL(blob) << "Unknown blob: " + blob_name;
    return blob->shape();
  }

  std::shared_ptr<Backend> &GetBackend() { return backend_; }

  const std::vector<std::string> &in_blob() const {
    return backend_->in_blob();
  }
  const std::vector<std::string> &out_blob() const {
    return backend_->out_blob();
  }

  bool has_argument(const std::string &name) const {
    return backend_->arg_helper().HasArgument(name);
  }
  template <typename T>
  T get_single_argument(const std::string &name, const T &default_value) const {
    return backend_->arg_helper().GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  std::vector<T> get_repeated_argument(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    return backend_->arg_helper().GetRepeatedArgument<T>(name, default_value);
  }

 private:
  Workspace ws_;

  std::shared_ptr<Backend> backend_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_IMPL_HPP
