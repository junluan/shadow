#ifndef SHADOW_DROPOUT_LAYER_H
#define SHADOW_DROPOUT_LAYER_H

#include "layer.hpp"

class DropoutLayer : public Layer {
public:
  explicit DropoutLayer(LayerType type);
  ~DropoutLayer();

  void MakeDropoutLayer(SizeParams params, float probability);
  void ForwardLayer();

#ifdef USE_CUDA
  void CUDAForwardLayer();
#endif

#ifdef USE_CL
  void CLForwardLayer();
#endif

  void ReleaseLayer();
};

#endif // SHADOW_DROPOUT_LAYER_H
