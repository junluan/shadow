#ifndef SHADOW_CORE_ALLOCATOR_HPP
#define SHADOW_CORE_ALLOCATOR_HPP

#include <cstdlib>

namespace Shadow {

enum class Device { kCPU, kGPU };

class Allocator {
 public:
  template <typename T>
  static void *MakeBuffer(size_t size, const void *host_ptr, int align = 1);

  template <typename T>
  static void ReadBuffer(size_t size, const void *src, void *des);

  template <typename T>
  static void WriteBuffer(size_t size, const void *src, void *des);

  template <typename T>
  static void CopyBuffer(size_t size, const void *src, void *des);

  template <typename T>
  static void ReleaseBuffer(void *device_ptr);
};

}  // namespace Shadow

#endif  // SHADOW_CORE_ALLOCATOR_HPP
