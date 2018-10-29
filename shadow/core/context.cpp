#include "context.hpp"

#include "kernel.hpp"
#include "util/log.hpp"

namespace Shadow {

Context::Context(int device_id) { Init(device_id); }

Context::~Context() { Clear(); }

void Context::Reset(int device_id) {
  Clear();
  Init(device_id);
}

void Context::SwitchDevice() {
#if defined(USE_CUDA)
  CUDA_CHECK(cudaSetDevice(device_id_));
#endif
}

void* Context::blas_handle() {
#if defined(USE_CUDA)
  CHECK_NOTNULL(blas_handle_);
  return blas_handle_;
#else
  return nullptr;
#endif
}

void* Context::cudnn_handle() {
#if defined(USE_CUDNN)
  CHECK_NOTNULL(cudnn_handle_);
  return cudnn_handle_;
#else
  return nullptr;
#endif
}

void* Context::nnpack_handle() {
#if defined(USE_NNPACK)
  CHECK_NOTNULL(nnpack_handle_);
  return nnpack_handle_;
#else
  return nullptr;
#endif
}

void Context::Init(int device_id) {
  device_id_ = device_id;

  SwitchDevice();

#if defined(USE_CUDA)
  if (blas_handle_ == nullptr) {
    CUBLAS_CHECK(cublasCreate((cublasHandle_t*)&blas_handle_));
    CHECK_NOTNULL(blas_handle_);
  }
#endif

#if defined(USE_CUDNN)
  if (cudnn_handle_ == nullptr) {
    CUDNN_CHECK(cudnnCreate((cudnnHandle_t*)&cudnn_handle_));
    CHECK_NOTNULL(cudnn_handle_);
  }
#endif

#if defined(USE_NNPACK)
  if (nnpack_handle_ == nullptr) {
    CHECK_EQ(nnp_initialize(), nnp_status_success);
    nnpack_handle_ = pthreadpool_create(0);
    CHECK_NOTNULL(nnpack_handle_);
  }
#endif
}

void Context::Clear() {
#if defined(USE_CUDA)
  if (blas_handle_ != nullptr) {
    CUBLAS_CHECK(cublasDestroy(cublasHandle_t(blas_handle_)));
    blas_handle_ = nullptr;
  }
#endif

#if defined(USE_CUDNN)
  if (cudnn_handle_ != nullptr) {
    CUDNN_CHECK(cudnnDestroy(cudnnHandle_t(cudnn_handle_)));
    cudnn_handle_ = nullptr;
  }
#endif

#if defined(USE_NNPACK)
  if (nnpack_handle_ != nullptr) {
    CHECK_EQ(nnp_deinitialize(), nnp_status_success);
    pthreadpool_destroy(pthreadpool_t(nnpack_handle_));
    nnpack_handle_ = nullptr;
  }
#endif
}

}  // namespace Shadow
