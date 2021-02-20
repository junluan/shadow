#ifndef SHADOW_CORE_EXTERNAL_HPP_
#define SHADOW_CORE_EXTERNAL_HPP_

#include "util/log.hpp"

#if defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#if defined(USE_CUDNN)
#include "cudnn.h"
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
inline void createTensorDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename T>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc, int n, int c, int h,
                            int w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc, CUDNN_TENSOR_NCHW,
                                         dataType<T>::type, n, c, h, w));
}

template <typename T>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc, int n,
                            const int* dim) {
  int stride[CUDNN_DIM_MAX] = {};
  stride[n - 1] = 1;
  for (int d = n - 2; d >= 0; --d) {
    stride[d] = dim[d + 1] * stride[d + 1];
  }
  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(*desc, dataType<T>::type, n, dim, stride));
}

template <typename T>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
}

template <typename T>
inline void setFilter4dDesc(cudnnFilterDescriptor_t* desc, int n, int c, int h,
                            int w) {
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<T>::type,
                                         CUDNN_TENSOR_NCHW, n, c, h, w));
}

template <typename T>
inline void setFilterNdDesc(cudnnFilterDescriptor_t* desc, int n,
                            const int* dim) {
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(*desc, dataType<T>::type,
                                         CUDNN_TENSOR_NCHW, n, dim));
}

template <typename T>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv_desc) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv_desc));
}

template <typename T>
inline void setConvolution2dDesc(cudnnConvolutionDescriptor_t* conv_desc,
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
inline void setConvolutionNdDesc(cudnnConvolutionDescriptor_t* conv_desc, int n,
                                 const int* pad, const int* stride,
                                 const int* dilation, int group) {
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(*conv_desc, n, pad, stride,
                                              dilation, CUDNN_CROSS_CORRELATION,
                                              dataType<T>::type));
#if CUDNN_VERSION_MIN(7, 0, 1)
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(*conv_desc, group));
#endif
}

template <typename T>
inline void createActivationDesc(cudnnActivationDescriptor_t* activate_desc) {
  CUDNN_CHECK(cudnnCreateActivationDescriptor(activate_desc));
}

template <typename T>
inline void setActivationDesc(cudnnActivationDescriptor_t* activate_desc,
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
    case 6:
      mode = CUDNN_ACTIVATION_CLIPPED_RELU;
      break;
    default:
      LOG(FATAL) << "Unsupported activate type " << activate_type;
  }
  CUDNN_CHECK(cudnnSetActivationDescriptor(*activate_desc, mode,
                                           CUDNN_NOT_PROPAGATE_NAN, coef));
}

template <typename T>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc) {
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
}

inline cudnnPoolingMode_t getPoolingMode(int pool_type) {
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
  return mode;
}

template <typename T>
inline void setPooling2dDesc(cudnnPoolingDescriptor_t* pool_desc, int pool_type,
                             int window_h, int window_w, int pad_h, int pad_w,
                             int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(
      *pool_desc, getPoolingMode(pool_type), CUDNN_PROPAGATE_NAN, window_h,
      window_w, pad_h, pad_w, stride_h, stride_w));
}

template <typename T>
inline void setPoolingNdDesc(cudnnPoolingDescriptor_t* pool_desc, int pool_type,
                             int n, const int* window, const int* pad,
                             const int* stride) {
  CUDNN_CHECK(cudnnSetPoolingNdDescriptor(*pool_desc, getPoolingMode(pool_type),
                                          CUDNN_PROPAGATE_NAN, n, window, pad,
                                          stride));
}

template <typename T>
inline void createReduceDesc(cudnnReduceTensorDescriptor_t* reduce_desc) {
  CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(reduce_desc));
}

template <typename T>
inline void setReduceDesc(cudnnReduceTensorDescriptor_t* reduce_desc,
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
    case 5:
      mode = CUDNN_REDUCE_TENSOR_NORM1;
      break;
    case 6:
      mode = CUDNN_REDUCE_TENSOR_NORM2;
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
    cudnnSpatialTransformerDescriptor_t* spatial_transformer_desc) {
  CUDNN_CHECK(
      cudnnCreateSpatialTransformerDescriptor(spatial_transformer_desc));
}

template <typename T>
inline void setSpatialTransformerDesc(
    cudnnSpatialTransformerDescriptor_t* spatial_transformer_desc, int n,
    const int* dim) {
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

inline dnnl::memory::format_tag get_memory_format(int num_axes) {
  switch (num_axes) {
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    case 6:
      return dnnl::memory::format_tag::abcdef;
    case 7:
      return dnnl::memory::format_tag::abcdefg;
    case 8:
      return dnnl::memory::format_tag::abcdefgh;
    case 9:
      return dnnl::memory::format_tag::abcdefghi;
    case 10:
      return dnnl::memory::format_tag::abcdefghij;
    case 11:
      return dnnl::memory::format_tag::abcdefghijk;
    case 12:
      return dnnl::memory::format_tag::abcdefghijkl;
    default:
      return dnnl::memory::format_tag::undef;
  }
}

template <typename T>
inline dnnl::memory::desc create_memory_desc(
    const std::vector<int>& shape, dnnl::memory::format_tag format_tag) {
  auto data_type = dnnl::memory::data_type::undef;
  if (std::is_same<T, std::int32_t>::value) {
    data_type = dnnl::memory::data_type::s32;
  } else if (std::is_same<T, std::int8_t>::value) {
    data_type = dnnl::memory::data_type::s8;
  } else if (std::is_same<T, std::uint8_t>::value) {
    data_type = dnnl::memory::data_type::u8;
  } else if (std::is_same<T, float>::value) {
    data_type = dnnl::memory::data_type::f32;
  }
  return dnnl::memory::desc(dnnl::memory::dims(shape.begin(), shape.end()),
                            data_type, format_tag);
}

inline void batch_normalization_forward(
    void* dnnl_handle, const dnnl::batch_normalization_forward::desc& desc,
    const void* src_data, const void* mean_data, const void* variance_data,
    void* dst_data) {
  auto* stream = static_cast<dnnl::stream*>(dnnl_handle);
  const auto& engine = stream->get_engine();
  const auto& primitive_desc =
      dnnl::batch_normalization_forward::primitive_desc(desc, engine);
  const auto& src_mem = dnnl::memory(primitive_desc.src_desc(), engine,
                                     const_cast<void*>(src_data));
  const auto& mean_mem = dnnl::memory(primitive_desc.mean_desc(), engine,
                                      const_cast<void*>(mean_data));
  const auto& variance_mem = dnnl::memory(
      primitive_desc.variance_desc(), engine, const_cast<void*>(variance_data));
  const auto& dst_mem =
      dnnl::memory(primitive_desc.dst_desc(), engine, dst_data);
  dnnl::batch_normalization_forward(primitive_desc)
      .execute(*stream, {{DNNL_ARG_SRC, src_mem},
                         {DNNL_ARG_MEAN, mean_mem},
                         {DNNL_ARG_VARIANCE, variance_mem},
                         {DNNL_ARG_DST, dst_mem}});
  stream->wait();
}

inline void binary_forward(void* dnnl_handle, const dnnl::binary::desc& desc,
                           const void* src_a_data, const void* src_b_data,
                           void* dst_data) {
  auto* stream = static_cast<dnnl::stream*>(dnnl_handle);
  const auto& engine = stream->get_engine();
  const auto& primitive_desc = dnnl::binary::primitive_desc(desc, engine);
  const auto& src_a_mem = dnnl::memory(primitive_desc.src0_desc(), engine,
                                       const_cast<void*>(src_a_data));
  const auto& src_b_mem = dnnl::memory(primitive_desc.src1_desc(), engine,
                                       const_cast<void*>(src_b_data));
  const auto& dst_mem =
      dnnl::memory(primitive_desc.dst_desc(), engine, dst_data);
  dnnl::binary(primitive_desc)
      .execute(*stream, {{DNNL_ARG_SRC_0, src_a_mem},
                         {DNNL_ARG_SRC_1, src_b_mem},
                         {DNNL_ARG_DST, dst_mem}});
  stream->wait();
}

inline void concat_forward(void* dnnl_handle,
                           const dnnl::concat::primitive_desc& primitive_desc,
                           const std::vector<const void*>& srcs_data,
                           void* dst_data) {
  auto* stream = static_cast<dnnl::stream*>(dnnl_handle);
  const auto& engine = stream->get_engine();
  std::unordered_map<int, dnnl::memory> args;
  for (int n = 0; n < srcs_data.size(); ++n) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + n,
                 dnnl::memory(primitive_desc.src_desc(n), engine,
                              const_cast<void*>(srcs_data[n]))});
  }
  args.insert({DNNL_ARG_DST,
               dnnl::memory(primitive_desc.dst_desc(), engine, dst_data)});
  dnnl::concat(primitive_desc).execute(*stream, args);
  stream->wait();
}

inline void matmul_forward(void* dnnl_handle, const dnnl::matmul::desc& desc,
                           const void* src_a_data, const void* src_b_data,
                           void* dst_data) {
  auto* stream = static_cast<dnnl::stream*>(dnnl_handle);
  const auto& engine = stream->get_engine();
  const auto& primitive_desc = dnnl::matmul::primitive_desc(desc, engine);
  const auto& src_mem = dnnl::memory(primitive_desc.src_desc(), engine,
                                     const_cast<void*>(src_a_data));
  const auto& weight_mem = dnnl::memory(primitive_desc.weights_desc(), engine,
                                        const_cast<void*>(src_b_data));
  const auto& dst_mem =
      dnnl::memory(primitive_desc.dst_desc(), engine, dst_data);
  dnnl::matmul(primitive_desc)
      .execute(*stream, {{DNNL_ARG_SRC, src_mem},
                         {DNNL_ARG_WEIGHTS, weight_mem},
                         {DNNL_ARG_DST, dst_mem}});
  stream->wait();
}

inline void reorder_forward(void* dnnl_handle,
                            const dnnl::reorder::primitive_desc& primitive_desc,
                            const void* src_data, void* dst_data) {
  auto* stream = static_cast<dnnl::stream*>(dnnl_handle);
  const auto& engine = stream->get_engine();
  const auto& src_mem = dnnl::memory(primitive_desc.src_desc(), engine,
                                     const_cast<void*>(src_data));
  const auto& dst_mem =
      dnnl::memory(primitive_desc.dst_desc(), engine, dst_data);
  dnnl::reorder(primitive_desc)
      .execute(*stream, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  stream->wait();
}

template <typename T>
inline void common_forward(void* dnnl_handle, const typename T::desc& desc,
                           const void* src_data, const void* weight_data,
                           const void* bias_data, void* dst_data,
                           int activate_type) {
  auto* stream = static_cast<dnnl::stream*>(dnnl_handle);
  const auto& engine = stream->get_engine();
  auto primitive_desc = typename T::primitive_desc(desc, engine);
  if (activate_type == 1) {
    dnnl::post_ops ops;
    ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 1.f);
    dnnl::primitive_attr attr;
    attr.set_post_ops(ops);
    primitive_desc = typename T::primitive_desc(desc, attr, engine);
  }
  const auto& src_mem = dnnl::memory(primitive_desc.src_desc(), engine,
                                     const_cast<void*>(src_data));
  const auto& weight_mem = dnnl::memory(primitive_desc.weights_desc(), engine,
                                        const_cast<void*>(weight_data));
  const auto& bias_mem = dnnl::memory(primitive_desc.bias_desc(), engine,
                                      const_cast<void*>(bias_data));
  const auto& dst_mem =
      dnnl::memory(primitive_desc.dst_desc(), engine, dst_data);
  T(primitive_desc)
      .execute(*stream, {{DNNL_ARG_SRC, src_mem},
                         {DNNL_ARG_WEIGHTS, weight_mem},
                         {DNNL_ARG_BIAS, bias_mem},
                         {DNNL_ARG_DST, dst_mem}});
  stream->wait();
}

template <typename T>
inline void common_forward(void* dnnl_handle, const typename T::desc& desc,
                           const void* src_data, void* dst_data) {
  auto* stream = static_cast<dnnl::stream*>(dnnl_handle);
  const auto& engine = stream->get_engine();
  const auto& primitive_desc = typename T::primitive_desc(desc, engine);
  const auto& src_mem = dnnl::memory(primitive_desc.src_desc(), engine,
                                     const_cast<void*>(src_data));
  const auto& dst_mem =
      dnnl::memory(primitive_desc.dst_desc(), engine, dst_data);
  T(primitive_desc)
      .execute(*stream, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  stream->wait();
}

}  // namespace idnnl

#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_EXTERNAL_HPP_
