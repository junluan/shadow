#include "allocator.hpp"

#include <cstring>

namespace Shadow {

#if !defined(USE_CUDA)

inline size_t align_size(int sz, int n) { return (sz + n - 1) & -n; }

template <typename T>
inline T *align_ptr(T *ptr, int n = sizeof(T)) {
  return (T *)(((size_t)ptr + n - 1) & -n);
}

inline void *fast_malloc(size_t size, int align) {
  auto *u_data = new unsigned char[size + sizeof(void *) + align]();
  unsigned char **a_data = align_ptr((unsigned char **)u_data + 1, align);
  a_data[-1] = u_data;
  return a_data;
}

inline void fast_free(void *ptr) {
  if (ptr != nullptr) {
    unsigned char *u_data = ((unsigned char **)ptr)[-1];
    delete[] u_data;
  }
}

template <typename T>
void *Allocator::MakeBuffer(size_t size, const void *host_ptr, int align) {
  auto *device_ptr = fast_malloc(size * sizeof(T), align);
  if (host_ptr != nullptr) {
    WriteBuffer<T>(size, host_ptr, device_ptr);
  }
  return device_ptr;
}

template <typename T>
void Allocator::ReadBuffer(size_t size, const void *src, void *des) {
  memcpy(des, src, size * sizeof(T));
}

template <typename T>
void Allocator::WriteBuffer(size_t size, const void *src, void *des) {
  memcpy(des, src, size * sizeof(T));
}

template <typename T>
void Allocator::CopyBuffer(size_t size, const void *src, void *des) {
  memcpy(des, src, size * sizeof(T));
}

template <typename T>
void Allocator::ReleaseBuffer(void *device_ptr) {
  fast_free(device_ptr);
}

#define INSTANTIATE_ALLOCATOR(T)                                            \
  template void *Allocator::MakeBuffer<T>(size_t, const void *, int align); \
  template void Allocator::ReadBuffer<T>(size_t, const void *, void *);     \
  template void Allocator::WriteBuffer<T>(size_t, const void *, void *);    \
  template void Allocator::CopyBuffer<T>(size_t, const void *, void *);     \
  template void Allocator::ReleaseBuffer<T>(void *);

INSTANTIATE_ALLOCATOR(int)
INSTANTIATE_ALLOCATOR(float)
INSTANTIATE_ALLOCATOR(unsigned char)
#undef INSTANTIATE_BUFFER_OPERATION

#endif

}  // namespace Shadow
