#ifndef SHADOW_CORE_EXTERNAL_HPP
#define SHADOW_CORE_EXTERNAL_HPP

#include "util/log.hpp"

#if defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#if defined(USE_CUDNN)
#include "cudnn.h"
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
#include "dnnl.hpp"
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

namespace Shadow {

#if defined(USE_CUDNN)

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition)                                             \
  do {                                                                     \
    cudnnStatus_t status = condition;                                      \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status); \
  } while (0)

namespace cudnn {

template <typename T>
class dataType;

template <>
class dataType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};

template <>
class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename T>
inline void createTensorDesc(cudnnTensorDescriptor_t *desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename T>
inline void setTensor4dDesc(cudnnTensorDescriptor_t *desc, int n, int c, int h,
                            int w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc, CUDNN_TENSOR_NCHW,
                                         dataType<T>::type, n, c, h, w));
}

template <typename T>
inline void setTensorNdDesc(cudnnTensorDescriptor_t *desc, int n, int *dim) {
  int stride[CUDNN_DIM_MAX] = {};
  stride[n - 1] = 1;
  for (int d = n - 2; d >= 0; --d) {
    stride[d] = dim[d + 1] * stride[d + 1];
  }
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(*desc, dataType<T>::type, n, dim, stride));
}

template <typename T>
inline void createFilterDesc(cudnnFilterDescriptor_t *desc) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
}

template <typename T>
inline void setFilter4dDesc(cudnnFilterDescriptor_t *desc, int n, int c, int h,
                            int w) {
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<T>::type,
                                         CUDNN_TENSOR_NCHW, n, c, h, w));
}

template <typename T>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t *conv_desc) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv_desc));
}

template <typename T>
inline void setConvolution2dDesc(cudnnConvolutionDescriptor_t *conv_desc,
                                 int pad_h, int pad_w, int stride_h,
                                 int stride_w, int dilation_h, int dilation_w,
                                 int group) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      *conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      CUDNN_CROSS_CORRELATION, dataType<T>::type));
#if CUDNN_VERSION_MIN(7, 0, 1)
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(*conv_desc, group));
#endif
}

template <typename T>
inline void createActivationDesc(cudnnActivationDescriptor_t *activate_desc) {
  CUDNN_CHECK(cudnnCreateActivationDescriptor(activate_desc));
}

template <typename T>
inline void setActivationDesc(cudnnActivationDescriptor_t *activate_desc,
                              int activate_type, double coef) {
  cudnnActivationMode_t mode{};
  switch (activate_type) {
#if CUDNN_VERSION_MIN(7, 1, 1)
    case -1:
      mode = CUDNN_ACTIVATION_IDENTITY;
      break;
#endif
    case 1:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case 3:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    case 5:
      mode = CUDNN_ACTIVATION_TANH;
      break;
    default:
      LOG(FATAL) << "Unsupported activate type " << activate_type;
  }
  CUDNN_CHECK(cudnnSetActivationDescriptor(*activate_desc, mode,
                                           CUDNN_NOT_PROPAGATE_NAN, coef));
}

template <typename T>
inline void createPoolingDesc(cudnnPoolingDescriptor_t *pool_desc) {
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
}

template <typename T>
inline void setPooling2dDesc(cudnnPoolingDescriptor_t *pool_desc, int pool_type,
                             int window_h, int window_w, int pad_h, int pad_w,
                             int stride_h, int stride_w) {
  cudnnPoolingMode_t mode{};
  switch (pool_type) {
    case 0:
      mode = CUDNN_POOLING_MAX;
      break;
    case 1:
      mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
    default:
      LOG(FATAL) << "Unsupported pool type " << pool_type;
  }
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, mode, CUDNN_PROPAGATE_NAN,
                                          window_h, window_w, pad_h, pad_w,
                                          stride_h, stride_w));
}

template <typename T>
inline void createReduceDesc(cudnnReduceTensorDescriptor_t *reduce_desc) {
  CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(reduce_desc));
}

template <typename T>
inline void setReduceDesc(cudnnReduceTensorDescriptor_t *reduce_desc,
                          int reduce_type) {
  cudnnReduceTensorOp_t mode{};
  switch (reduce_type) {
    case 0:
      mode = CUDNN_REDUCE_TENSOR_MUL;
      break;
    case 1:
      mode = CUDNN_REDUCE_TENSOR_ADD;
      break;
    case 2:
      mode = CUDNN_REDUCE_TENSOR_MAX;
      break;
    case 3:
      mode = CUDNN_REDUCE_TENSOR_MIN;
      break;
    case 4:
      mode = CUDNN_REDUCE_TENSOR_AVG;
      break;
    default:
      LOG(FATAL) << "Unsupported reduce type " << reduce_type;
  }
  CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
      *reduce_desc, mode, dataType<T>::type, CUDNN_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
}

template <typename T>
inline void createSpatialTransformerDesc(
    cudnnSpatialTransformerDescriptor_t *spatial_transformer_desc) {
  CUDNN_CHECK(
      cudnnCreateSpatialTransformerDescriptor(spatial_transformer_desc));
}

template <typename T>
inline void setSpatialTransformerDesc(
    cudnnSpatialTransformerDescriptor_t *spatial_transformer_desc, int n,
    int *dim) {
  CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(
      *spatial_transformer_desc, CUDNN_SAMPLER_BILINEAR,
      cudnn::dataType<T>::type, n, dim));
}

}  // namespace cudnn

#endif

}  // namespace Shadow

namespace Shadow {

#if defined(USE_DNNL)

namespace idnnl {

template <typename T>
inline dnnl::memory::desc create_memory_desc(
    const std::vector<int> &shape,
    dnnl::memory::format_tag format_tag = dnnl::memory::format_tag::nchw) {
  dnnl::memory::dims dims;
  for (auto dim : shape) {
    dims.push_back(dim);
  }
  auto data_type = dnnl::memory::data_type::undef;
  if (std::is_same<T, float>::value) {
    data_type = dnnl::memory::data_type::f32;
  } else if (std::is_same<T, int>::value) {
    data_type = dnnl::memory::data_type::s32;
  } else if (std::is_same<T, unsigned char>::value) {
    data_type = dnnl::memory::data_type::u8;
  }
  return dnnl::memory::desc(dims, data_type, format_tag);
}

dnnl::convolution_forward::desc create_convolution_desc(
    const dnnl::memory::desc &src_desc, const dnnl::memory::desc &weight_desc,
    const dnnl::memory::desc &bias_desc, const dnnl::memory::desc &dst_desc,
    int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h = 1,
    int dilation_w = 1);

void convolution_forward(void *dnnl_engine, void *dnnl_stream,
                         const dnnl::convolution_forward::desc &conv_desc,
                         const void *src_data, const void *weight_data,
                         const void *bias_data, void *dst_data,
                         int activate_type = -1);

}  // namespace idnnl

#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_EXTERNAL_HPP
