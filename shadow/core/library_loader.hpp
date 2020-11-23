#ifndef SHADOW_CORE_LIBRARY_LOADER_HPP_
#define SHADOW_CORE_LIBRARY_LOADER_HPP_

#include <map>
#include <string>

namespace Shadow {

class LibraryLoader {
 public:
  ~LibraryLoader();

  static std::string GetRuntimeLibraryPath();

  static std::string GetSharedName(const std::string& library_name);

  bool Load(const std::string& library_path);

 private:
  static void* LoadShared(const std::string& library_path);

  std::map<std::string, void*> libraries_;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_LIBRARY_LOADER_HPP_
