#ifndef SHADOW_OPERATORS_POOLING_OP_HPP
#define SHADOW_OPERATORS_POOLING_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PoolingOp : public Operator {
 public:
  PoolingOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    pool_type_ = get_single_argument<int>("pool", 0);
    global_pooling_ = get_single_argument<bool>("global_pooling", false);
    if (!global_pooling_) {
      const auto &kernel_size = get_repeated_argument<int>("kernel_size");
      CHECK_LE(kernel_size.size(), 2);
      if (kernel_size.empty()) {
        kernel_size_h_ = kernel_size_w_ =
            get_single_argument<int>("kernel_size", 0);
      } else if (kernel_size.size() == 1) {
        kernel_size_h_ = kernel_size_w_ = kernel_size[0];
      } else {
        kernel_size_h_ = kernel_size[0], kernel_size_w_ = kernel_size[1];
      }
      const auto &stride = get_repeated_argument<int>("stride");
      CHECK_LE(stride.size(), 2);
      if (stride.empty()) {
        stride_h_ = stride_w_ = get_single_argument<int>("stride", 1);
      } else if (stride.size() == 1) {
        stride_h_ = stride_w_ = stride[0];
      } else {
        stride_h_ = stride[0], stride_w_ = stride[1];
      }
      const auto &pad = get_repeated_argument<int>("pad");
      CHECK_LE(pad.size(), 2);
      if (pad.empty()) {
        pad_h_ = pad_w_ = get_single_argument<int>("pad", 0);
      } else if (pad.size() == 1) {
        pad_h_ = pad_w_ = pad[0];
      } else {
        pad_h_ = pad[0], pad_w_ = pad[1];
      }
    }
    full_pooling_ = get_single_argument<bool>("full_pooling", true);

#if defined(USE_CUDNN)
    cudnn::createPoolingDesc<float>(&pooling_desc_);
    cudnn::createTensorDesc<float>(&bottom_desc_);
    cudnn::createTensorDesc<float>(&top_desc_);
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

  void Forward() override;

 private:
  int pool_type_, kernel_size_h_, kernel_size_w_, stride_h_, stride_w_, pad_h_,
      pad_w_;
  bool global_pooling_, full_pooling_;

#if defined(USE_CUDNN)
  cudnnPoolingDescriptor_t pooling_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
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
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size_h,
             int kernel_size_w, int stride_h, int stride_w, int pad_h,
             int pad_w, int mode, const VecInt &out_shape, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_POOLING_OP_HPP
