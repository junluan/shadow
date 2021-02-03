#include "workspace.hpp"

namespace Shadow {

Workspace::Workspace(const ArgumentHelper& arguments) {
#if defined(USE_CUDA)
  const auto& backend_type =
      arguments.GetSingleArgument<std::string>("backend_type", "");
  if (backend_type == "Native" || backend_type == "TensorRT") {
    context_ = GetContext<DeviceType::kGPU>(arguments);
  } else {
    context_ = GetContext<DeviceType::kCPU>(arguments);
  }
#else
  context_ = GetContext<DeviceType::kCPU>(arguments);
#endif
}

std::shared_ptr<Context>& Workspace::Ctx() { return context_; }

bool Workspace::HasBlob(const std::string& name) const {
  return static_cast<bool>(blob_map_.count(name));
}

DataType Workspace::GetBlobDataType(const std::string& name) const {
  CHECK(HasBlob(name)) << "Unknown blob: " + name;
  return blob_map_.at(name)->data_type();
}

std::vector<int> Workspace::GetBlobShape(const std::string& name) const {
  CHECK(HasBlob(name)) << "Unknown blob: " + name;
  return blob_map_.at(name)->shape();
}

std::shared_ptr<Blob> Workspace::GetBlob(const std::string& name) const {
  if (HasBlob(name)) {
    return blob_map_.at(name);
  } else {
    DLOG(WARNING) << "Blob " << name << " not in the workspace.";
    return nullptr;
  }
}

std::shared_ptr<Blob> Workspace::CreateBlob(const std::string& name,
                                            DataType data_type,
                                            Allocator* allocator) {
  if (!HasBlob(name)) {
    blob_map_[name] = std::make_shared<Blob>(
        name, data_type,
        (allocator == nullptr) ? context_->allocator() : allocator);
  }
  return GetBlob(name);
}

std::shared_ptr<Blob> Workspace::CreateBlob(const std::string& name,
                                            const std::string& data_type_str,
                                            Allocator* allocator) {
  if (data_type_str == "int64") {
    return CreateBlob(name, DataType::kI64, allocator);
  } else if (data_type_str == "int32") {
    return CreateBlob(name, DataType::kI32, allocator);
  } else if (data_type_str == "int16") {
    return CreateBlob(name, DataType::kI16, allocator);
  } else if (data_type_str == "int8") {
    return CreateBlob(name, DataType::kI8, allocator);
  } else if (data_type_str == "uint64") {
    return CreateBlob(name, DataType::kU64, allocator);
  } else if (data_type_str == "uint32") {
    return CreateBlob(name, DataType::kU32, allocator);
  } else if (data_type_str == "uint16") {
    return CreateBlob(name, DataType::kU16, allocator);
  } else if (data_type_str == "uint8") {
    return CreateBlob(name, DataType::kU8, allocator);
  } else if (data_type_str == "float") {
    return CreateBlob(name, DataType::kF32, allocator);
  } else if (data_type_str == "bool") {
    return CreateBlob(name, DataType::kBool, allocator);
  } else {
    return nullptr;
  }
}

std::shared_ptr<Blob> Workspace::CreateTempBlob(const std::vector<int>& shape,
                                                DataType data_type) {
  auto count =
      std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                      std::multiplies<size_t>());
  CHECK_GT(count, 0);
  auto temp_blob =
      std::make_shared<Blob>("temp", data_type, context_->allocator());
  temp_blob->share_data(GetTempPtr(count * temp_blob->elem_size()), shape);
  return temp_blob;
}

void Workspace::GrowTempBuffer(size_t raw_size) {
  auto temp_blob = CreateBlob("temp_blob", DataType::kI32);
  size_t num_int = raw_size / temp_blob->elem_size() + 1;
  CHECK_LE(num_int, std::numeric_limits<int>::max());
  temp_blob->reshape({static_cast<int>(num_int)});
  temp_offset_ = 0;
}

size_t Workspace::GetWorkspaceSize() const {
  size_t mem_size = 0;
  for (const auto& blob_it : blob_map_) {
    mem_size += blob_it.second->max_size();
  }
  return mem_size;
}

size_t Workspace::GetWorkspaceTempSize() const {
  const auto temp_blob = GetBlob("temp_blob");
  return temp_blob != nullptr ? temp_blob->max_size() : 0;
}

const void* Workspace::GetTempPtr(size_t raw_size) {
  const auto temp_blob = GetBlob("temp_blob");
  CHECK_NOTNULL(temp_blob);
  CHECK_LE(temp_offset_ + raw_size, temp_blob->max_size());
  const auto* ptr = temp_blob->data<unsigned char>() + temp_offset_;
  temp_offset_ += raw_size;
  return ptr;
}

}  // namespace Shadow
