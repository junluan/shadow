#ifndef SHADOW_LAYERS_DROPOUT_LAYER_HPP
#define SHADOW_LAYERS_DROPOUT_LAYER_HPP

#include "shadow/layers/layer.hpp"

class DropoutLayer : public Layer {
public:
  explicit DropoutLayer(shadow::LayerParameter layer_param);
  ~DropoutLayer();

  void MakeLayer(Blob<BType> *blob);

  void ForwardLayer();

  void ReleaseLayer();
};

#endif // SHADOW_LAYERS_DROPOUT_LAYER_HPP
