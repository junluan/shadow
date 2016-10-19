#ifndef SHADOW_LAYERS_POOLING_LAYER_HPP
#define SHADOW_LAYERS_POOLING_LAYER_HPP

#include "shadow/layers/layer.hpp"

class PoolingLayer : public Layer {
 public:
  PoolingLayer() {}
  explicit PoolingLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~PoolingLayer() { Release(); }

  void Reshape();
  void Forward();
  void Release();

 private:
  int kernel_size_, stride_;
  shadow::PoolingParameter::PoolType pool_type_;
};

#endif  // SHADOW_LAYERS_POOLING_LAYER_HPP
