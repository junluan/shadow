#ifndef SHADOW_CORE_KERNEL_HPP
#define SHADOW_CORE_KERNEL_HPP

#if defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudnn.hpp"
#endif

#if defined(USE_NNPACK)
#include "nnpack.h"
#endif

#if defined(USE_DNNL)
#include "idnnl.hpp"
#endif

#include <cstdlib>

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

namespace Kernel {

void Synchronize();

template <typename T, typename Dtype>
T *MakeBuffer(size_t size, Dtype *host_ptr);

template <typename T, typename Dtype>
void ReadBuffer(size_t size, const T *src, Dtype *des);

template <typename T, typename Dtype>
void WriteBuffer(size_t size, const Dtype *src, T *des);

template <typename T, typename Dtype = float>
void CopyBuffer(size_t size, const T *src, T *des);

template <typename T>
void ReleaseBuffer(T *buffer);

}  // namespace Kernel

}  // namespace Shadow

#endif  // SHADOW_CORE_KERNEL_HPP
