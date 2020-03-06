#ifndef SHADOW_CORE_ALLOCATOR_HPP
#define SHADOW_CORE_ALLOCATOR_HPP

#include <cstdlib>

namespace Shadow {

enum class DeviceType { kCPU, kGPU };

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual DeviceType device_type() const = 0;

  virtual void *malloc(size_t size, const void *host_ptr) const = 0;

  virtual void read(size_t size, const void *src, void *dst) const = 0;

  virtual void write(size_t size, const void *src, void *dst) const = 0;

  virtual void copy(size_t size, const void *src, void *dst) const = 0;

  virtual void free(void *ptr) const = 0;
};

template <DeviceType D>
Allocator *GetAllocator();

}  // namespace Shadow

#endif  // SHADOW_CORE_ALLOCATOR_HPP
