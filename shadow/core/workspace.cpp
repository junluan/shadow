#include "workspace.hpp"

namespace Shadow {

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
  if (blob_temp_ == nullptr) {
    blob_temp_ = std::make_shared<Blob<unsigned char>>("temp_buffer");
  }
  blob_temp_->reshape({count, elem_size});
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

void Workspace::CreateCtx(int device_id) {
  if (context_ == nullptr) {
    context_ = std::make_shared<Context>(device_id);
  } else {
    context_->Reset(device_id);
  }
}

Context *Workspace::Ctx() {
  CHECK_NOTNULL(context_);
  return context_.get();
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

void *Workspace::GetTempPtr(size_t count, int elem_size) {
  auto required = count * elem_size;
  CHECK_LE(temp_offset_ + required, blob_temp_->mem_count());
  auto *ptr = blob_temp_->mutable_data() + temp_offset_;
  temp_offset_ += required;
  return ptr;
}

}  // namespace Shadow
