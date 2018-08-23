#include "workspace.hpp"

namespace Shadow {

bool Workspace::RemoveBlob(const std::string &name) {
  auto blob_it = blob_map_.find(name);
  if (blob_it != blob_map_.end()) {
    const auto &blob_type = blob_map_.at(name).first;
    auto *blob = blob_map_.at(name).second;
    ClearBlob(blob_type, blob);
    blob_map_.erase(blob_it);
    return true;
  }
  DLOG(WARNING) << "Blob " << name << " not exists. Skipping.";
  return false;
}

int Workspace::GetWorkspaceSize() const {
  int count = 0;
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

int Workspace::GetWorkspaceTempSize() const {
  int count = 0;
  for (const auto &blob_it : blob_temp_) {
    count += blob_it->mem_count();
  }
  return count;
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
  int sufficient_id = -1;
  for (int n = 0; n < blob_temp_.size(); ++n) {
    if (required <= blob_temp_[n]->mem_count()) {
      sufficient_id = n;
      break;
    }
  }
  if (sufficient_id == -1) {
    sufficient_id = static_cast<int>(blob_temp_.size());
    blob_temp_.push_back(new Blob<unsigned char>(VecInt{count, size}));
    DLOG(INFO) << "New temp buffer allocated: " << (required >> 20) << " MB.";
  }
  return blob_temp_[sufficient_id]->mutable_data();
}

}  // namespace Shadow
