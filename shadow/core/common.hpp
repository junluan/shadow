#ifndef SHADOW_CORE_COMMON_HPP
#define SHADOW_CORE_COMMON_HPP

#if defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudnn.hpp"
#endif

#if defined(USE_Eigen)
#include "Eigen/Eigen"
template <typename T>
using MapVector = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using MapMatrix = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

#if defined(USE_NNPACK)
#include "nnpack.h"
#endif

#if defined(USE_DNNL)
#include "idnnl.hpp"
#endif

#define SHADOW_STRINGIFY_IMPL(s) #s
#define SHADOW_STRINGIFY(s) SHADOW_STRINGIFY_IMPL(s)

#define SHADOW_CONCATENATE_IMPL(s1, s2) s1##s2
#define SHADOW_CONCATENATE(s1, s2) SHADOW_CONCATENATE_IMPL(s1, s2)

#ifdef __COUNTER__
#define SHADOW_ANONYMOUS_VARIABLE(s) SHADOW_CONCATENATE(s, __COUNTER__)
#else
#define SHADOW_ANONYMOUS_VARIABLE(s) SHADOW_CONCATENATE(s, __LINE__)
#endif

#define SHADOW_VERSION_MAJOR 0
#define SHADOW_VERSION_MINOR 1
#define SHADOW_VERSION_PATCH 0
#define SHADOW_VERSION_STRING \
  SHADOW_STRINGIFY(           \
      SHADOW_VERSION_MAJOR.SHADOW_VERSION_MINOR.SHADOW_VERSION_PATCH)

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&) = delete;    \
  classname& operator=(const classname&) = delete

namespace Shadow {

#if defined(USE_CUDA)

// CUDA: use 512 threads per block
const int NumThreads = 512;

// CUDA: number of blocks for threads
inline int GetBlocks(const int N) { return (N + NumThreads - 1) / NumThreads; }

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition)                                  \
  do {                                                         \
    cudaError_t error = condition;                             \
    CHECK_EQ(error, cudaSuccess) << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition)              \
  do {                                       \
    cublasStatus_t status = condition;       \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS); \
  } while (0)

#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_COMMON_HPP
