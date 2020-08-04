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

class StaticLinkingProtector {
 public:
  StaticLinkingProtector() {
    const auto& registered_backends = BackendRegistry()->Keys();
    CHECK(!registered_backends.empty())
        << "You might have made a build error: the Shadow library does not "
           "seem to be linked with whole-static library option. To do so, use "
           "-Wl,-force_load (clang) or -Wl,--whole-archive (gcc) to link the "
           "Shadow library.";
  }
};

std::shared_ptr<Backend> CreateBackend(const ArgumentHelper& arguments,
                                       Workspace* ws) {
  static StaticLinkingProtector g_protector;
  CHECK(arguments.HasArgument("backend_type"));
  const auto& backend_type =
      arguments.GetSingleArgument<std::string>("backend_type", "");
  auto backend = std::shared_ptr<Backend>(
      BackendRegistry()->Create(backend_type, arguments, ws));
  CHECK_NOTNULL(backend)
      << "Backend type: " << backend_type
      << " is not registered, currently registered backend types: "
      << Util::format_vector(BackendRegistry()->Keys(), ", ", "[", "]");
  return backend;
}

SHADOW_DEFINE_REGISTRY(BackendRegistry, Backend, const ArgumentHelper&,
                       Workspace*);

}  // namespace Shadow
