#ifndef SHADOW_DROPOUT_LAYER_H
#define SHADOW_DROPOUT_LAYER_H

#include "layer.h"

class DropoutLayer : public Layer {
public:
  DropoutLayer(LayerType type);
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
