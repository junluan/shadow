#ifndef SHADOW_LAYERS_POOLING_LAYER_HPP
#define SHADOW_LAYERS_POOLING_LAYER_HPP

#include "shadow/layers/layer.hpp"

#include <string>

class PoolingLayer : public Layer {
public:
  explicit PoolingLayer(shadow::LayerParameter layer_param);
  ~PoolingLayer();

  void MakeLayer(shadow::BlobShape *shape);
  void ForwardLayer();

#ifdef USE_CUDA
  void CUDAForwardLayer();
#endif

#ifdef USE_CL
  void CLForwardLayer();
#endif

  void ReleaseLayer();

  shadow::PoolType pool_type_;
  int kernel_size_, stride_;
};

#endif // SHADOW_LAYERS_POOLING_LAYER_HPP
