#ifndef SHADOW_CORE_WORKSPACE_HPP
#define SHADOW_CORE_WORKSPACE_HPP

#include "blob.hpp"

#include <map>
#include <string>
#include <typeinfo>

namespace Shadow {

static const std::string int_id(typeid(int).name());
static const std::string float_id(typeid(float).name());
static const std::string uchar_id(typeid(unsigned char).name());

class Workspace {
 public:
  Workspace() = default;
  ~Workspace() {
    for (auto blob_it : blob_map_) {
      const auto &blob_type = blob_it.second.first;
      auto *blob = blob_it.second.second;
      ClearBlob(blob_type, blob);
    }
    blob_map_.clear();
  }

  bool HasBlob(const std::string &name) const {
    return static_cast<bool>(blob_map_.count(name));
  }

  template <typename T>
  Blob<T> *CreateBlob(const std::string &name) {
    return CreateBlob<T>({}, name);
  }
  template <typename T>
  Blob<T> *CreateBlob(const VecInt &shape, const std::string &name,
                      bool shared = false) {
    if (HasBlob(name)) {
      DLOG(WARNING) << "Blob " << name << " already exists. Skipping.";
    } else {
      blob_map_[name].first = typeid(T).name();
      if (!shape.empty()) {
        blob_map_[name].second = new Blob<T>(shape, name, shared);
      } else {
        blob_map_[name].second = new Blob<T>(name);
      }
    }
    return GetBlob<T>(name);
  }
  bool RemoveBlob(const std::string &name);

  template <typename T>
  const Blob<T> *GetBlob(const std::string &name) const {
    if (blob_map_.count(name)) {
      const auto &blob_type = blob_map_.at(name).first;
      const auto ask_type = typeid(T).name();
      CHECK(blob_type == ask_type) << "Blob " << name << " has type "
                                   << blob_type << ", but ask for " << ask_type;
      return static_cast<const Blob<T> *>(blob_map_.at(name).second);
    }
    DLOG(WARNING) << "Blob " << name << " not in the workspace.";
    return nullptr;
  }
  template <typename T>
  Blob<T> *GetBlob(const std::string &name) {
    return const_cast<Blob<T> *>(
        static_cast<const Workspace *>(this)->GetBlob<T>(name));
  }

  const std::string GetBlobType(const std::string &name) const {
    if (blob_map_.count(name)) {
      return blob_map_.at(name).first;
    }
    DLOG(WARNING) << "Blob " << name << " not in the workspace.";
    return std::string();
  }

 private:
  void ClearBlob(const std::string &blob_type, void *blob);

  std::map<std::string, std::pair<std::string, void *>> blob_map_;

  DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace Shadow

#endif  // SHADOW_CORE_WORKSPACE_HPP
