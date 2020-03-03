#ifndef SHADOW_CORE_ALLOCATOR_HPP
#define SHADOW_CORE_ALLOCATOR_HPP

#include <cstdlib>

namespace Shadow {

enum class Device { kCPU, kGPU };

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual Device GetDevice() const = 0;

  virtual void *MakeBuffer(size_t size, const void *host_ptr) const = 0;

  virtual void ReadBuffer(size_t size, const void *src, void *dst) const = 0;

  virtual void WriteBuffer(size_t size, const void *src, void *dst) const = 0;

  virtual void CopyBuffer(size_t size, const void *src, void *dst) const = 0;

  virtual void ReleaseBuffer(void *ptr) const = 0;
};

template <Device D>
Allocator *GetAllocator();

}  // namespace Shadow

#endif  // SHADOW_CORE_ALLOCATOR_HPP
