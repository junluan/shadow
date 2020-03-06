#ifndef SHADOW_OPERATORS_CONV_OP_HPP
#define SHADOW_OPERATORS_CONV_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ConvOp : public Operator {
 public:
  ConvOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    num_output_ = get_single_argument<int>("num_output", 0);
    CHECK(has_argument("kernel_size"));
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
    dilation_ = get_single_argument<int>("dilation", 1);
    group_ = get_single_argument<int>("group", 1);
    CHECK_EQ(num_output_ % group_, 0);
    bias_term_ = get_single_argument<bool>("bias_term", true);
    activate_type_ = get_single_argument<int>("type", -1);
    CHECK((activate_type_ == -1 || activate_type_ == 1))
        << "Build in activate only support Relu";

#if defined(USE_CUDNN)
#if CUDNN_VERSION_MIN(7, 0, 1)
    use_cudnn_ = true;
#else
    use_cudnn_ = group_ == 1;
#endif
    if (use_cudnn_) {
      cudnn::createConvolutionDesc<float>(&conv_desc_);
      cudnn::createTensorDesc<float>(&bottom_desc_);
      cudnn::createTensorDesc<float>(&top_desc_);
      cudnn::createFilterDesc<float>(&filter_desc_);
      if (bias_term_) {
        cudnn::createTensorDesc<float>(&bias_desc_);
      }
      if (activate_type_ == 1) {
        cudnn::createActivationDesc<float>(&activate_desc_);
      }
    }
#endif
  }
  ~ConvOp() override {
#if defined(USE_CUDNN)
    if (conv_desc_ != nullptr) {
      cudnnDestroyConvolutionDescriptor(conv_desc_);
      conv_desc_ = nullptr;
    }
    if (bottom_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bottom_desc_);
      bottom_desc_ = nullptr;
    }
    if (top_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(top_desc_);
      top_desc_ = nullptr;
    }
    if (filter_desc_ != nullptr) {
      cudnnDestroyFilterDescriptor(filter_desc_);
      filter_desc_ = nullptr;
    }
    if (bias_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bias_desc_);
      bias_desc_ = nullptr;
    }
    if (activate_desc_ != nullptr) {
      cudnnDestroyActivationDescriptor(activate_desc_);
      activate_desc_ = nullptr;
    }
#endif
  }

  void Forward() override;

 private:
  int num_output_, kernel_size_h_, kernel_size_w_, stride_h_, stride_w_, pad_h_,
      pad_w_, dilation_, group_, activate_type_, out_spatial_dim_, kernel_dim_;
  int weight_offset_, col_offset_, output_offset_;
  bool bias_term_, use_cudnn_ = false, use_nnpack_ = false,
                   use_depthwise_ = false;

#if defined(USE_CUDNN)
  cudnnConvolutionFwdAlgo_t fwd_algo_ =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
  cudnnFilterDescriptor_t filter_desc_ = nullptr;
  cudnnTensorDescriptor_t bias_desc_ = nullptr;

  cudnnActivationDescriptor_t activate_desc_ = nullptr;
#endif

#if defined(USE_NNPACK)
  nnp_convolution_algorithm nnp_algorithm_ = nnp_convolution_algorithm_auto;
  nnp_convolution_transform_strategy nnp_transform_ =
      nnp_convolution_transform_strategy_compute;
  nnp_activation nnp_activation_ = nnp_activation_identity;
  nnp_size nnp_input_size_, nnp_kernel_size_, nnp_stride_;
  nnp_padding nnp_pad_;
#endif
};

static inline int conv_out_size(int dim, int kernel_size, int stride, int pad,
                                int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return (dim + 2 * pad - kernel_extent) / stride + 1;
}

namespace Vision {

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
            int pad_h, int pad_w, int dilation, int zero_point,
            const VecInt &out_shape, T *col_data, Context *context);

template <typename T>
void Depthwise(const T *in_data, const VecInt &in_shape, const T *weight_data,
               const T *bias_data, int kernel_size_h, int kernel_size_w,
               int stride_h, int stride_w, int pad_h, int pad_w, int dilation,
               int bias_term, const VecInt &out_shape, T *out_data,
               Context *context);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_CONV_OP_HPP
