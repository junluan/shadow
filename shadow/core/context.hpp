#ifndef SHADOW_CORE_CONTEXT_HPP
#define SHADOW_CORE_CONTEXT_HPP

#include "allocator.hpp"
#include "helper.hpp"

namespace Shadow {

class Context {
 public:
  virtual ~Context() = default;

  virtual Allocator* allocator() const = 0;

  virtual DeviceType device_type() const = 0;
  virtual int device_id() const = 0;

  virtual void switch_device() = 0;
  virtual void synchronize() = 0;

  virtual void* blas_handle() const { return nullptr; }
  virtual void* cudnn_handle() const { return nullptr; }
  virtual void* nnpack_handle() const { return nullptr; }
  virtual void* dnnl_engine() const { return nullptr; }
  virtual void* dnnl_stream() const { return nullptr; }
};

template <DeviceType D>
std::shared_ptr<Context> GetContext(const ArgumentHelper& arguments);

}  // namespace Shadow

#endif  // SHADOW_CORE_CONTEXT_HPP
