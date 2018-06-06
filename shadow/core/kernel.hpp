#ifndef SHADOW_CORE_KERNEL_HPP
#define SHADOW_CORE_KERNEL_HPP

#if !defined(USE_CUDA) & !defined(USE_CL)
#if defined(USE_NNPACK)
#include "nnpack.h"
#endif

#elif defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudnn.hpp"

#elif defined(USE_CL)
#include "util/easycl.hpp"
#endif

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

#define CUDA_CHECK(condition)                                         \
  do {                                                                \
    cudaError_t error = condition;                                    \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#endif

#if defined(USE_CL)
using BufferI = EasyCL::Buffer<int>;
using BufferF = EasyCL::Buffer<float>;
using BufferUC = EasyCL::Buffer<unsigned char>;
#endif

namespace Kernel {

void Setup(int device_id = 0);
void Release();

void Synchronize();

template <typename T, typename Dtype>
T *MakeBuffer(int size, Dtype *host_ptr);

template <typename T, typename Dtype>
void ReadBuffer(int size, const T *src, Dtype *des);

template <typename T, typename Dtype>
void WriteBuffer(int size, const Dtype *src, T *des);

template <typename T, typename Dtype = float>
void CopyBuffer(int size, const T *src, T *des);

template <typename T>
void ReleaseBuffer(T *buffer);

#if !defined(USE_CUDA) & !defined(USE_CL)
#if defined(USE_NNPACK)
const int NumThreads = 0;

extern pthreadpool_t nnp_pthreadpool_;
#endif

#elif defined(USE_CUDA)
extern cublasHandle_t cublas_handle_;

#elif defined(USE_CL)
extern EasyCL::Device *device_;
extern EasyCL::Context *context_;
extern EasyCL::Queue *queue_;
extern EasyCL::Event *event_;

extern EasyCL::KernelSet cl_kernels_;
#endif

}  // namespace Kernel

}  // namespace Shadow

#endif  // SHADOW_CORE_KERNEL_HPP
