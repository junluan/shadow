#ifndef SHADOW_OPERATORS_CONV_OP_HPP
#define SHADOW_OPERATORS_CONV_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ConvOp : public Operator {
 public:
  explicit ConvOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    num_output_ = get_single_argument<int>("num_output", 0);
    CHECK(has_argument("kernel_size"));
    kernel_size_ = get_single_argument<int>("kernel_size", 0);
    stride_ = get_single_argument<int>("stride", 1);
    pad_ = get_single_argument<int>("pad", 0);
    dilation_ = get_single_argument<int>("dilation", 1);
    group_ = get_single_argument<int>("group", 1);
    CHECK_EQ(bottoms<float>(0)->shape(1) % group_, 0);
    CHECK_EQ(num_output_ % group_, 0);
    bias_term_ = get_single_argument<bool>("bias_term", true);
    activate_type_ = get_single_argument<int>("type", -1);
    CHECK((activate_type_ == -1 || activate_type_ == 1))
        << "Build in activate only support Relu";

    if (bias_term_) {
      CHECK_EQ(blobs_size(), 2);
    } else {
      CHECK_EQ(blobs_size(), 1);
    }

#if defined(USE_CUDNN)
    use_cudnn_ = dilation_ == 1 && group_ == 1;
    if (use_cudnn_) {
      cudnn::createConvolutionDesc<float>(&conv_desc_);
      cudnn::createTensor4dDesc<float>(&bottom_desc_);
      cudnn::createTensor4dDesc<float>(&top_desc_);
      cudnn::createFilterDesc<float>(&filter_desc_, num_output_,
                                     bottoms<float>(0)->shape(1), kernel_size_,
                                     kernel_size_);
      if (bias_term_) {
        cudnn::createTensor4dDesc<float>(&bias_desc_);
        cudnn::setTensor4dDesc<float>(&bias_desc_, 1, num_output_, 1, 1);
      }
    }
#endif

#if defined(USE_NNPACK)
    use_nnpack_ = bottoms<float>(0)->shape(0) == 1 && group_ == 1 &&
                  dilation_ == 1 && bias_term_;
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

    if (workspace_ != nullptr) {
      cudaFree(workspace_);
      workspace_ = nullptr;
    }
#endif
  }

  void Reshape() override;
  void Forward() override;

 protected:
  int num_output_, kernel_size_, stride_, pad_, dilation_, group_,
      activate_type_, out_spatial_dim_, kernel_dim_;
  int weight_offset_, col_offset_, output_offset_;
  bool bias_term_, use_cudnn_ = false, use_nnpack_ = false;

  BlobF *biases_multiplier_ = nullptr, *col_image_ = nullptr;

#if defined(USE_CUDNN)
  cudnnConvolutionFwdAlgo_t fwd_algo_ =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
  cudnnFilterDescriptor_t filter_desc_ = nullptr;
  cudnnTensorDescriptor_t bias_desc_ = nullptr;

  size_t workspace_fwd_size_ = 0;
  void *workspace_ = nullptr;
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
            int kernel_size, int stride, int pad, int dilation, int zero_point,
            const VecInt &out_shape, T *col_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_CONV_OP_HPP
