#ifndef SHADOW_CORE_COMMON_HPP
#define SHADOW_CORE_COMMON_HPP

#include <cstdlib>

#if defined(USE_Eigen)
#include "Eigen/Eigen"

template <typename T>
using MapVector = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using MapMatrix = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

#define SHADOW_VERSION_MAJOR 0
#define SHADOW_VERSION_MINOR 1
#define SHADOW_VERSION_PATCH 0
#define SHADOW_VERSION                                         \
  (SHADOW_VERSION_MAJOR * 10000 + SHADOW_VERSION_MINOR * 100 + \
   SHADOW_VERSION_PATCH)

namespace Shadow {

#define MALLOC_ALIGN 16

static inline int align_size(int sz, int n) { return (sz + n - 1) & -n; }

template <typename T>
static inline T* align_ptr(T* ptr, int n = sizeof(T)) {
  return (T*)(((size_t)ptr + n - 1) & -n);
}

static inline void* fast_malloc(int size, int align) {
  auto* u_data = new unsigned char[size + sizeof(void*) + align]();
  unsigned char** a_data = align_ptr((unsigned char**)u_data + 1, align);
  a_data[-1] = u_data;
  return a_data;
}

static inline void fast_free(void* ptr) {
  if (ptr != nullptr) {
    unsigned char* u_data = ((unsigned char**)ptr)[-1];
    delete[] u_data;
  }
}

#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&) = delete;    \
  classname& operator=(const classname&) = delete
#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_COMMON_HPP
