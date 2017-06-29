#include "workspace.hpp"

namespace Shadow {

BlobF *Workspace::CreateBlob(const std::string &name) {
  if (HasBlob(name)) {
    DLOG(WARNING) << "Blob " << name << " already exists. Skipping.";
  } else {
    blob_map_[name] = new BlobF(name);
  }
  return GetBlob(name);
}

BlobF *Workspace::CreateBlob(const VecInt &shape, const std::string &name,
                             bool shared) {
  if (HasBlob(name)) {
    DLOG(WARNING) << "Blob " << name << " already exists. Skipping.";
  } else {
    blob_map_[name] = new BlobF(shape, name, shared);
  }
  return GetBlob(name);
}

bool Workspace::RemoveBlob(const std::string &name) {
  auto it = blob_map_.find(name);
  if (it != blob_map_.end()) {
    blob_map_.erase(it);
    return true;
  }
  DLOG(WARNING) << "Blob " << name << " not exists. Skipping.";
  return false;
}

const BlobF *Workspace::GetBlob(const std::string &name) const {
  if (blob_map_.count(name)) {
    return blob_map_.at(name);
  } else {
    DLOG(WARNING) << "Blob " << name << " not in the workspace.";
    return nullptr;
  }
}

BlobF *Workspace::GetBlob(const std::string &name) {
  return const_cast<BlobF *>(
      static_cast<const Workspace *>(this)->GetBlob(name));
}

}  // namespace Shadow
