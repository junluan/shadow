#ifndef SHADOW_LAYERS_CONV_LAYER_HPP
#define SHADOW_LAYERS_CONV_LAYER_HPP

#include "shadow/layers/layer.hpp"

class ConvLayer : public Layer {
 public:
  ConvLayer() {}
  explicit ConvLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ConvLayer() { Release(); }

  void Reshape();
  void Forward();
  void Release();

 private:
  int num_output_, kernel_size_, stride_, pad_, out_spatial_dim_, kernel_dim_;

  Blob<float> biases_multiplier_, col_image_;
};

#endif  // SHADOW_LAYERS_CONV_LAYER_HPP
