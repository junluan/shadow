#ifndef SHADOW_LAYERS_CONVOLUTION_LAYER_HPP
#define SHADOW_LAYERS_CONVOLUTION_LAYER_HPP

#include "shadow/layer.hpp"

class ConvolutionLayer : public Layer {
 public:
  ConvolutionLayer() {}
  explicit ConvolutionLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ConvolutionLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int num_output_, kernel_size_, stride_, pad_, dilation_, out_spatial_dim_,
      kernel_dim_;
  bool bias_term_;

  Blob<float> biases_multiplier_, col_image_;

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

#endif  // SHADOW_LAYERS_CONVOLUTION_LAYER_HPP
