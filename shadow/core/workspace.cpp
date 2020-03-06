#include "workspace.hpp"

namespace Shadow {

Workspace::Workspace(const ArgumentHelper &arguments) {
#if defined(USE_CUDA)
  context_ = GetContext<DeviceType::kGPU>(arguments);
#else
  context_ = GetContext<DeviceType::kCPU>(arguments);
#endif
}

Context *Workspace::Ctx() {
  CHECK_NOTNULL(context_);
  return context_.get();
}

bool Workspace::HasBlob(const std::string &name) const {
  return static_cast<bool>(blob_map_.count(name));
}

std::string Workspace::GetBlobType(const std::string &name) const {
  if (blob_map_.count(name)) {
    return blob_map_.at(name).first;
  }
  DLOG(WARNING) << "Blob " << name << " not in the workspace.";
  return std::string();
}

std::vector<int> Workspace::GetBlobShape(const std::string &name) const {
  if (blob_map_.count(name)) {
    const auto &blob_type = blob_map_.at(name).first;
    if (blob_type == int_id) {
      return static_cast<const BlobI *>(blob_map_.at(name).second)->shape();
    } else if (blob_type == float_id) {
      return static_cast<const BlobF *>(blob_map_.at(name).second)->shape();
    } else if (blob_type == uchar_id) {
      return static_cast<const BlobUC *>(blob_map_.at(name).second)->shape();
    } else {
      LOG(FATAL) << "Unknown blob type " << blob_type;
    }
  }
  DLOG(WARNING) << "Blob " << name << " not in the workspace.";
  return std::vector<int>();
}

void Workspace::GrowTempBuffer(int count, int elem_size) {
  auto *temp_blob = CreateBlob<unsigned char>("temp_blob");
  CHECK_NOTNULL(temp_blob);
  temp_blob->reshape({count, elem_size});
  temp_offset_ = 0;
}

size_t Workspace::GetWorkspaceSize() const {
  size_t count = 0;
  for (const auto &blob_it : blob_map_) {
    const auto &blob_type = blob_it.second.first;
    auto *blob = blob_it.second.second;
    if (blob != nullptr) {
      if (blob_type == int_id) {
        count += static_cast<BlobI *>(blob)->mem_count();
      } else if (blob_type == float_id) {
        count += static_cast<BlobF *>(blob)->mem_count();
      } else if (blob_type == uchar_id) {
        count += static_cast<BlobUC *>(blob)->mem_count();
      } else {
        LOG(FATAL) << "Unknown blob type " << blob_type;
      }
    }
  }
  return count;
}

size_t Workspace::GetWorkspaceTempSize() const {
  const auto *temp_blob = GetBlob<unsigned char>("temp_blob");
  return temp_blob != nullptr ? temp_blob->mem_count() : 0;
}

void Workspace::ClearBlob(const std::string &blob_type, void *blob) {
  if (blob != nullptr) {
    if (blob_type == int_id) {
      delete static_cast<BlobI *>(blob);
    } else if (blob_type == float_id) {
      delete static_cast<BlobF *>(blob);
    } else if (blob_type == uchar_id) {
      delete static_cast<BlobUC *>(blob);
    } else {
      LOG(FATAL) << "Unknown blob type " << blob_type;
    }
    blob = nullptr;
  }
}

const void *Workspace::GetTempPtr(size_t count, int elem_size) {
  const auto *temp_blob = GetBlob<unsigned char>("temp_blob");
  CHECK_NOTNULL(temp_blob);
  auto required = count * elem_size;
  CHECK_LE(temp_offset_ + required, temp_blob->mem_count());
  const auto *ptr = temp_blob->data() + temp_offset_;
  temp_offset_ += required;
  return ptr;
}

}  // namespace Shadow
