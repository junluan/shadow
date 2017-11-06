#ifndef SHADOW_OPERATORS_POOLING_OP_HPP
#define SHADOW_OPERATORS_POOLING_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PoolingOp : public Operator {
 public:
  explicit PoolingOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    pool_type_ = get_single_argument<int>("pool", 0);
    global_pooling_ = get_single_argument<bool>("global_pooling", false);
    if (!global_pooling_) {
      CHECK(has_argument("kernel_size"));
      kernel_size_ = get_single_argument<int>("kernel_size", 2);
      stride_ = get_single_argument<int>("stride", 1);
      pad_ = get_single_argument<int>("pad", 0);
    } else {
      kernel_size_ = bottoms<float>(0)->shape(2);
      stride_ = 1;
      pad_ = 0;
    }
    full_pooling_ = get_single_argument<bool>("full_pooling", true);

#if defined(USE_CUDNN)
    cudnn::createPoolingDesc<float>(&pooling_desc_, pool_type_, &mode_,
                                    kernel_size_, kernel_size_, pad_, pad_,
                                    stride_, stride_);
    cudnn::createTensor4dDesc<float>(&bottom_desc_);
    cudnn::createTensor4dDesc<float>(&top_desc_);
#endif
  }
  ~PoolingOp() override {
#if defined(USE_CUDNN)
    if (pooling_desc_ != nullptr) {
      cudnnDestroyPoolingDescriptor(pooling_desc_);
      pooling_desc_ = nullptr;
    }
    if (bottom_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bottom_desc_);
      bottom_desc_ = nullptr;
    }
    if (top_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(top_desc_);
      top_desc_ = nullptr;
    }
#endif
  }

  void Reshape() override;
  void Forward() override;

 private:
  int pool_type_, kernel_size_, stride_, pad_;
  bool global_pooling_, full_pooling_;

#if defined(USE_CUDNN)
  cudnnPoolingDescriptor_t pooling_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
  cudnnPoolingMode_t mode_;
#endif
};

static inline int pooling_out_size(int dim, int kernel_size, int stride,
                                   int pad, bool full_pooling = true) {
  if (full_pooling) {
    return static_cast<int>(
        std::ceil((dim + 2.f * pad - kernel_size) / stride) + 1);
  } else {
    return static_cast<int>(
        std::floor((dim + 2.f * pad - kernel_size) / stride) + 1);
  }
}

namespace Vision {

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_POOLING_OP_HPP
