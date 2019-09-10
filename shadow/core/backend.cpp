#include "backend.hpp"

#include "util/util.hpp"

namespace Shadow {

Backend *CreateBackend(const ArgumentHelper &arguments, Workspace *ws) {
  CHECK(arguments.HasArgument("backend_type"));
  const auto &backend_type =
      arguments.GetSingleArgument<std::string>("backend_type", "");
  auto *backend = BackendRegistry()->Create(backend_type, arguments, ws);
  LOG_IF(FATAL, backend == nullptr)
      << "Backend type: " << backend_type
      << " is not registered, currently registered backend types: "
      << Util::format_vector(BackendRegistry()->Keys());
  return backend;
}

SHADOW_DEFINE_REGISTRY(BackendRegistry, Backend, const ArgumentHelper &,
                       Workspace *);

}  // namespace Shadow
