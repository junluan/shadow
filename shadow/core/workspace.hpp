#ifndef SHADOW_CORE_WORKSPACE_HPP_
#define SHADOW_CORE_WORKSPACE_HPP_

#include "blob.hpp"
#include "context.hpp"

#include <map>
#include <memory>
#include <string>

namespace Shadow {

class Workspace {
 public:
  explicit Workspace(const ArgumentHelper& arguments);

  Context* Ctx();

  bool HasBlob(const std::string& name) const;

  DataType GetBlobDataType(const std::string& name) const;
  std::vector<int> GetBlobShape(const std::string& name) const;

  std::shared_ptr<Blob> GetBlob(const std::string& name) const;

  std::shared_ptr<Blob> CreateBlob(const std::string& name, DataType data_type,
                                   Allocator* allocator = nullptr);

  std::shared_ptr<Blob> CreateBlob(const std::string& name,
                                   const std::string& data_type_str,
                                   Allocator* allocator = nullptr);

  std::shared_ptr<Blob> CreateTempBlob(const std::vector<int>& shape,
                                       DataType data_type);

  void GrowTempBuffer(size_t raw_size);

  size_t GetWorkspaceSize() const;
  size_t GetWorkspaceTempSize() const;

 private:
  const void* GetTempPtr(size_t raw_size);

  std::shared_ptr<Context> context_{nullptr};

  std::map<std::string, std::shared_ptr<Blob>> blob_map_;

  size_t temp_offset_{0};

  DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace Shadow

#endif  // SHADOW_CORE_WORKSPACE_HPP_
