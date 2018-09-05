#include "workspace.hpp"

namespace Shadow {

bool Workspace::HasBlob(const std::string &name) const {
  return static_cast<bool>(blob_map_.count(name));
}

const std::string Workspace::GetBlobType(const std::string &name) const {
  if (blob_map_.count(name)) {
    return blob_map_.at(name).first;
  }
  DLOG(WARNING) << "Blob " << name << " not in the workspace.";
  return std::string();
}

void Workspace::GrowTempBuffer(int size) {
  if (blob_temp_ == nullptr) {
    blob_temp_ = std::make_shared<Blob<unsigned char>>(VecInt{size});
  } else {
    if (size > blob_temp_->mem_count()) {
      blob_temp_->reshape({size});
    }
  }
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
  return blob_temp_->mem_count();
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

void *Workspace::GetTempPtr(int count, int size) {
  auto required = static_cast<size_t>(count) * size;
  CHECK_LE(temp_offset_ + required, blob_temp_->mem_count());
  auto *ptr = blob_temp_->mutable_data() + temp_offset_;
  temp_offset_ += required;
  return ptr;
}

}  // namespace Shadow
