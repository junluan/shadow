#include "library_loader.hpp"

#include "util/log.hpp"
#include "util/util.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Shadow {

LibraryLoader::~LibraryLoader() {
  for (auto& it : libraries_) {
#if defined(_WIN32)
    CHECK(FreeLibrary(static_cast<HMODULE>(it.second)));
#else
    CHECK_EQ(dlclose(it.second), 0);
#endif
  }
}

std::string LibraryLoader::GetRuntimeLibraryPath() {
#if defined(_WIN32)
  return std::string();
#else
  Dl_info dl_info;
  CHECK_GT(dladdr(reinterpret_cast<void*>(GetRuntimeLibraryPath), &dl_info), 0);
  return Path(dl_info.dli_fname).parent_path().str();
#endif
}

std::string LibraryLoader::GetSharedName(const std::string& library_name) {
#if defined(_WIN32)
  return library_name + ".dll";
#elif defined(__APPLE__)
  return "lib" + library_name + ".dylib";
#else
  return "lib" + library_name + ".so";
#endif
}

bool LibraryLoader::Load(const std::string& library_path) {
  if (!libraries_.count(library_path)) {
    auto library = LoadShared(library_path);
    if (library != nullptr) {
      libraries_.insert({library_path, library});
    } else {
      return false;
    }
  }
  return true;
}

void* LibraryLoader::LoadShared(const std::string& library_path) {
#if defined(_WIN32)
  std::wstring library_path_w(library_path.begin(), library_path.end());
  auto library = LoadLibraryW(library_path_w.c_str());
  LOG_IF(WARNING, library == nullptr)
      << "Can not load library " << library_path;
#else
  auto library = dlopen(library_path.c_str(), RTLD_LAZY);
  LOG_IF(WARNING, library == nullptr)
      << "Can not load library " << library_path << ": " << dlerror();
#endif
  return library;
}

}  // namespace Shadow
