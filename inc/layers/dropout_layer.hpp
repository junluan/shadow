#ifndef SHADOW_DROPOUT_LAYER_HPP
#define SHADOW_DROPOUT_LAYER_HPP

#include "layer.hpp"

class DropoutLayer : public Layer {
public:
  explicit DropoutLayer(shadow::LayerParameter layer_param);
  ~DropoutLayer();

  void MakeLayer(shadow::BlobShape *shape);
  void ForwardLayer();

#ifdef USE_CUDA
  void CUDAForwardLayer();
#endif

#ifdef USE_CL
  void CLForwardLayer();
#endif

  void ReleaseLayer();
};

#endif // SHADOW_DROPOUT_LAYER_HPP
