#include "allocator.hpp"

#include <cstring>

namespace Shadow {

#if defined(__ANDROID__) || defined(ANDROID)
const int MemoryAlignment = 64;
#else
const int MemoryAlignment = 32;
#endif

inline size_t align_size(int sz, int n) { return (sz + n - 1) & -n; }

template <typename T>
inline T* align_ptr(T* ptr, int n = sizeof(T)) {
  return reinterpret_cast<T*>(((size_t)ptr + n - 1) & -n);
}

inline void* fast_malloc(size_t size, int align) {
  auto* u_data = new unsigned char[size + sizeof(void*) + align]();
  unsigned char** a_data = align_ptr((unsigned char**)u_data + 1, align);
  a_data[-1] = u_data;
  return a_data;
}

inline void fast_free(void* ptr) {
  if (ptr != nullptr) {
    unsigned char* u_data = ((unsigned char**)ptr)[-1];
    delete[] u_data;
  }
}

class CPUAllocator : public Allocator {
 public:
  DeviceType device_type() const override { return DeviceType::kCPU; }

  void* malloc(size_t size, const void* host_ptr) const override {
    auto* ptr = fast_malloc(size, MemoryAlignment);
    if (host_ptr != nullptr) {
      write(size, host_ptr, ptr);
    }
    return ptr;
  }

  void read(size_t size, const void* src, void* dst) const override {
    memcpy(dst, src, size);
  }

  void write(size_t size, const void* src, void* dst) const override {
    memcpy(dst, src, size);
  }

  void copy(size_t size, const void* src, void* dst) const override {
    memcpy(dst, src, size);
  }

  void free(void* ptr) const override { fast_free(ptr); }

  void set_stream(void* stream) override {}
};

template <>
std::shared_ptr<Allocator> GetAllocator<DeviceType::kCPU>() {
  return std::make_shared<CPUAllocator>();
}

}  // namespace Shadow
