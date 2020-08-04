#ifndef SHADOW_CORE_NETWORK_IMPL_HPP_
#define SHADOW_CORE_NETWORK_IMPL_HPP_

#include "backend.hpp"

namespace Shadow {

class NetworkImpl {
 public:
  void LoadXModel(const shadow::NetParam& net_param,
                  const ArgumentHelper& arguments) {
    ws_ = std::make_shared<Workspace>(arguments);
    backend_ = CreateBackend(arguments, ws_.get());
    backend_->LoadModel(net_param);
  }

  void Forward(const std::map<std::string, void*>& data_map,
               const std::map<std::string, std::vector<int>>& shape_map) {
    ws_->Ctx()->switch_device();

    CHECK_NOTNULL(backend_);
    backend_->Forward(data_map, shape_map);

    ws_->Ctx()->synchronize();
  }

  template <typename T>
  const T* GetBlobDataByName(const std::string& blob_name,
                             const std::string& locate) {
    auto blob = ws_->GetBlob(blob_name);
    CHECK_NOTNULL(blob) << "Unknown blob: " + blob_name;
    if (locate == "host") {
      return blob->cpu_data<T>();
    } else {
      return blob->data<T>();
    }
  }

  std::vector<int> GetBlobShapeByName(const std::string& blob_name) const {
    return ws_->GetBlobShape(blob_name);
  }

  std::shared_ptr<Backend>& GetBackend() { return backend_; }

  const std::vector<std::string>& in_blob() const {
    return backend_->in_blob();
  }
  const std::vector<std::string>& out_blob() const {
    return backend_->out_blob();
  }

  bool has_argument(const std::string& name) const {
    return backend_->arg_helper().HasArgument(name);
  }
  template <typename T>
  T get_single_argument(const std::string& name, const T& default_value) const {
    return backend_->arg_helper().GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  std::vector<T> get_repeated_argument(
      const std::string& name, const std::vector<T>& default_value = {}) const {
    return backend_->arg_helper().GetRepeatedArgument<T>(name, default_value);
  }

 private:
  std::shared_ptr<Workspace> ws_ = nullptr;
  std::shared_ptr<Backend> backend_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_NETWORK_IMPL_HPP_
