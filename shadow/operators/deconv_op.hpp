#ifndef SHADOW_OPERATORS_DECONV_OP_HPP
#define SHADOW_OPERATORS_DECONV_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DeconvOp : public Operator {
 public:
  DeconvOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    num_output_ = get_single_argument<int>("num_output", 0);
    CHECK(has_argument("kernel_size"));
    kernel_size_ = get_single_argument<int>("kernel_size", 0);
    stride_ = get_single_argument<int>("stride", 1);
    pad_ = get_single_argument<int>("pad", 0);
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
  ~DeconvOp() override {
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
  int num_output_, kernel_size_, stride_, pad_, dilation_, group_,
      activate_type_, conv_out_spatial_dim_, out_spatial_dim_, kernel_dim_;
  int conv_in_c, conv_out_c, weight_offset_, col_offset_, output_offset_;
  bool bias_term_, use_cudnn_ = false;

#if defined(USE_CUDNN)
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_ =
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

  cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
  cudnnFilterDescriptor_t filter_desc_ = nullptr;
  cudnnTensorDescriptor_t bias_desc_ = nullptr;

  cudnnActivationDescriptor_t activate_desc_ = nullptr;
#endif
};

static inline int deconv_out_size(int dim, int kernel_size, int stride, int pad,
                                  int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return stride * (dim - 1) + kernel_extent - 2 * pad;
}

namespace Vision {

template <typename T>
void Col2Im(const T *col_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation,
            const VecInt &out_shape, T *in_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DECONV_OP_HPP
