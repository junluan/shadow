#ifndef SHADOW_CORE_KERNEL_HPP
#define SHADOW_CORE_KERNEL_HPP

#if defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cudnn.hpp"

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

#elif defined(USE_CL)
#include "easycl/easycl.hpp"
typedef EasyCL::Buffer<int> BufferI;
typedef EasyCL::Buffer<float> BufferF;
#endif

namespace Kernel {

void Setup(int device_id = 0);
void Release();

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

#if defined(USE_CUDA)
extern cublasHandle_t cublas_handle_;

#elif defined(USE_CL)
extern EasyCL::Device *device_;
extern EasyCL::Context *context_;
extern EasyCL::Queue *queue_;
extern EasyCL::Event *event_;

extern EasyCL::KernelSet cl_kernels_;
#endif

}  // namespace Kernel

#endif  // SHADOW_CORE_KERNEL_HPP
