#include "backend.hpp"

#include "library_loader.hpp"

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

std::shared_ptr<Backend> CreateBackend(const ArgumentHelper& arguments) {
  static StaticLinkingProtector g_protector;
  CHECK(arguments.HasArgument("backend_type"));
  const auto& backend_type =
      arguments.GetSingleArgument<std::string>("backend_type", "");
  CHECK(!backend_type.empty());
  if (!BackendRegistry()->Has(backend_type)) {
    auto backend_library =
        arguments.GetSingleArgument<std::string>("backend_library", "");
    if (backend_library.empty()) {
      auto backend_type_lower = backend_type;
      std::transform(backend_type.begin(), backend_type.end(),
                     backend_type_lower.begin(), ::tolower);
      backend_library =
          LibraryLoader::GetRuntimeLibraryPath() + "/" +
          LibraryLoader::GetSharedName("backend_" + backend_type_lower);
    }
    static auto library_loader = LibraryLoader();
    CHECK(library_loader.Load(backend_library))
        << "Can not load backend library: " << backend_library;
  }
  auto backend = std::shared_ptr<Backend>(
      BackendRegistry()->Create(backend_type, arguments));
  CHECK_NOTNULL(backend)
      << "Backend type: " << backend_type
      << " is not registered, currently registered backend types: "
      << Util::format_vector(BackendRegistry()->Keys(), ", ", "[", "]");
  return backend;
}

SHADOW_DEFINE_REGISTRY(BackendRegistry, Backend, const ArgumentHelper&);

}  // namespace Shadow
