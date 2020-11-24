#ifndef SHADOW_CORE_CONTEXT_HPP_
#define SHADOW_CORE_CONTEXT_HPP_

#include "allocator.hpp"
#include "helper.hpp"

#include <memory>

namespace Shadow {

class Context {
 public:
  virtual ~Context() = default;

  virtual Allocator* allocator() const = 0;

  virtual DeviceType device_type() const = 0;
  virtual int device_id() const = 0;

  virtual void switch_device() = 0;
  virtual void synchronize() = 0;

  virtual void* stream() const { return nullptr; }

  virtual void* cublas_handle() const { return nullptr; }
  virtual void* cudnn_handle() const { return nullptr; }
  virtual void* nnpack_handle() const { return nullptr; }
  virtual void* dnnl_handle() const { return nullptr; }
};

template <DeviceType D>
std::shared_ptr<Context> GetContext(const ArgumentHelper& arguments);

}  // namespace Shadow

#endif  // SHADOW_CORE_CONTEXT_HPP_
