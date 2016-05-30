#ifndef SHADOW_POOLING_LAYER_HPP
#define SHADOW_POOLING_LAYER_HPP

#include "layer.hpp"

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

#endif // SHADOW_POOLING_LAYER_HPP
