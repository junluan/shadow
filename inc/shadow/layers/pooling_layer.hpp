#ifndef SHADOW_LAYERS_POOLING_LAYER_HPP
#define SHADOW_LAYERS_POOLING_LAYER_HPP

#include "shadow/layers/layer.hpp"

class PoolingLayer : public Layer {
public:
  explicit PoolingLayer(shadow::LayerParameter layer_param);
  ~PoolingLayer();

  void MakeLayer(Blob<BType> *blob);

  void ForwardLayer();

  void ReleaseLayer();

private:
  shadow::PoolType pool_type_;
  int kernel_size_, stride_;
};

#endif // SHADOW_LAYERS_POOLING_LAYER_HPP
