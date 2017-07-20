#ifndef SHADOW_OPERATORS_CONV_OP_HPP
#define SHADOW_OPERATORS_CONV_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ConvOp : public Operator {
 public:
  explicit ConvOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ConvOp() { Release(); }

  virtual void Setup() override;
  virtual void Reshape() override;
  virtual void Forward() override;
  virtual void Release() override;

 protected:
  int num_output_, kernel_size_, stride_, pad_, dilation_, group_,
      out_spatial_dim_, kernel_dim_;
  int weight_offset_, col_offset_, output_offset_;
  bool bias_term_, use_cudnn_ = false, use_nnpack_ = false;

  BlobF biases_multiplier_, col_image_;

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
  nnp_size nnp_input_size_, nnp_kernel_size_, nnp_stride_;
  nnp_padding nnp_pad_;
#endif
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_CONV_OP_HPP
