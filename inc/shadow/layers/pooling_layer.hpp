#ifndef SHADOW_LAYERS_POOLING_LAYER_HPP
#define SHADOW_LAYERS_POOLING_LAYER_HPP

#include "shadow/layers/layer.hpp"

class PoolingLayer : public Layer {
public:
  PoolingLayer() {}
  explicit PoolingLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~PoolingLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Forward();
  void Release();

private:
  shadow::PoolType pool_type_;
  int kernel_size_, stride_;
};

#endif // SHADOW_LAYERS_POOLING_LAYER_HPP
