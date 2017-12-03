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
    for (auto blob_it : blob_temp_) {
      delete blob_it;
      blob_it = nullptr;
    }
    blob_temp_.clear();
  }

  bool HasBlob(const std::string &name) const {
    return static_cast<bool>(blob_map_.count(name));
  }

  const std::string GetBlobType(const std::string &name) const {
    if (blob_map_.count(name)) {
      return blob_map_.at(name).first;
    }
    DLOG(WARNING) << "Blob " << name << " not in the workspace.";
    return std::string();
  }

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

  template <typename T>
  Blob<T> *CreateBlob(const std::string &name) {
    return CreateBlob<T>({}, name);
  }
  template <typename T>
  Blob<T> *CreateBlob(const VecInt &shape, const std::string &name,
                      bool shared = false) {
    if (HasBlob(name)) {
      if (!shape.empty()) {
        CHECK(shape == GetBlob<T>(name)->shape()) << "Blob " << name
                                                  << " shape mismatch";
      }
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

  template <typename Dtype>
  Blob<Dtype> *CreateTempBlob(const VecInt &shape, const std::string &name) {
    Blob<Dtype> *blob = nullptr;
    if (!HasBlob(name)) {
      blob_map_[name].first = typeid(Dtype).name();
      blob_map_[name].second = new Blob<Dtype>(name);
    }
    blob = GetBlob<Dtype>(name);
    CHECK_NOTNULL(blob);
    blob->clear();
    blob->set_shape(shape);
    int cou = 1;
    for (const auto dim : shape) cou *= dim;
    int required = cou * sizeof(Dtype) / sizeof(unsigned char);
    blob->share_data(reinterpret_cast<BACKEND *>(GetTempPtr(required)), shape);
    return blob;
  }

  bool RemoveBlob(const std::string &name);

  int GetWorkspaceSize() const;
  int GetWorkspaceTempSize() const;

 private:
  void ClearBlob(const std::string &blob_type, void *blob);

  void *GetTempPtr(int required);

  std::map<std::string, std::pair<std::string, void *>> blob_map_;
  std::vector<BlobUC *> blob_temp_;

  DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace Shadow

#endif  // SHADOW_CORE_WORKSPACE_HPP
