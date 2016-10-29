#ifndef SHADOW_LAYERS_CONVOLUTION_LAYER_HPP
#define SHADOW_LAYERS_CONVOLUTION_LAYER_HPP

#include "shadow/layers/layer.hpp"

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
};

#endif  // SHADOW_LAYERS_CONVOLUTION_LAYER_HPP
