#ifndef SHADOW_OPERATORS_CONVOLUTION_OP_HPP
#define SHADOW_OPERATORS_CONVOLUTION_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ConvolutionOp : public Operator {
 public:
  explicit ConvolutionOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ConvolutionOp() { Release(); }

  void Setup();
  void Reshape();
  void Forward();
  void Release();

 private:
  int num_output_, kernel_size_, stride_, pad_, dilation_, group_,
      out_spatial_dim_, kernel_dim_;
  int weight_offset_, col_offset_, output_offset_;
  bool bias_term_, use_cudnn_ = false;

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
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_CONVOLUTION_OP_HPP
