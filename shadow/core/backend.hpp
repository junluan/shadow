#ifndef SHADOW_CORE_BACKEND_HPP_
#define SHADOW_CORE_BACKEND_HPP_

#include "helper.hpp"
#include "registry.hpp"
#include "workspace.hpp"

#include "util/queue.hpp"

namespace Shadow {

class Backend {
 public:
  virtual ~Backend() = default;

  virtual void LoadModel(const shadow::NetParam& net_param) = 0;

  virtual void Run(
      const std::map<std::string, void*>& data_map,
      const std::map<std::string, std::vector<int>>& shape_map) = 0;

  virtual void SaveEngine(
      const std::string& save_path,
      std::map<std::string, std::vector<char>>* save_data) = 0;

  const ArgumentHelper& arg_helper() const { return arg_helper_; }

  const std::shared_ptr<Workspace>& ws() const { return ws_; }

  const std::vector<std::string>& in_blob() const { return in_blob_; }
  const std::vector<std::string>& out_blob() const { return out_blob_; }

  static Queue<std::shared_ptr<std::vector<char>>>& data_exchange_queue(
      const std::string& key, unsigned int max_size = 16);

 protected:
  ArgumentHelper arg_helper_;

  std::shared_ptr<Workspace> ws_ = nullptr;
  std::vector<std::string> in_blob_, out_blob_;
};

std::shared_ptr<Backend> CreateBackend(const ArgumentHelper& arguments);

SHADOW_DECLARE_REGISTRY(BackendRegistry, Backend, const ArgumentHelper&);

#define REGISTER_BACKEND(name, ...) \
  SHADOW_REGISTER_CLASS(BackendRegistry, name, __VA_ARGS__)

}  // namespace Shadow

#endif  // SHADOW_CORE_BACKEND_HPP_
