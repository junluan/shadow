#ifndef SHADOW_CORE_WORKSPACE_HPP
#define SHADOW_CORE_WORKSPACE_HPP

#include "blob.hpp"

#include <map>

namespace Shadow {

class Workspace {
 public:
  typedef std::map<std::string, BlobF *> BlobMap;

  Workspace() {}
  ~Workspace() {
    for (auto it : blob_map_) {
      if (it.second != nullptr) {
        delete it.second;
        it.second = nullptr;
      }
    }
    blob_map_.clear();
  }

  bool HasBlob(const std::string &name) const {
    return static_cast<bool>(blob_map_.count(name));
  }

  BlobF *CreateBlob(const std::string &name);
  BlobF *CreateBlob(const VecInt &shape, const std::string &name,
                    bool shared = false);
  bool RemoveBlob(const std::string &name);

  const BlobF *GetBlob(const std::string &name) const;
  BlobF *GetBlob(const std::string &name);

 private:
  BlobMap blob_map_;
};

}  // namespace Shadow

#endif  // SHADOW_CORE_WORKSPACE_HPP
