#include "backend.hpp"

#include "util/util.hpp"

namespace Shadow {

Queue<std::shared_ptr<std::vector<char>>>& Backend::data_exchange_queue(
    const std::string& key, unsigned int max_size) {
  static std::map<std::string,
                  std::shared_ptr<Queue<std::shared_ptr<std::vector<char>>>>>
      g_queue;
  if (!g_queue.count(key)) {
    g_queue[key] =
        std::make_shared<Queue<std::shared_ptr<std::vector<char>>>>(max_size);
  }
  return *g_queue.at(key);
}

Backend* CreateBackend(const ArgumentHelper& arguments, Workspace* ws) {
  CHECK(arguments.HasArgument("backend_type"));
  const auto& backend_type =
      arguments.GetSingleArgument<std::string>("backend_type", "");
  auto* backend = BackendRegistry()->Create(backend_type, arguments, ws);
  LOG_IF(FATAL, backend == nullptr)
      << "Backend type: " << backend_type
      << " is not registered, currently registered backend types: "
      << Util::format_vector(BackendRegistry()->Keys(), ", ", "[", "]");
  return backend;
}

SHADOW_DEFINE_REGISTRY(BackendRegistry, Backend, const ArgumentHelper&,
                       Workspace*);

}  // namespace Shadow
