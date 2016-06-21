#ifndef SHADOW_LAYERS_DROPOUT_LAYER_HPP
#define SHADOW_LAYERS_DROPOUT_LAYER_HPP

#include "shadow/layers/layer.hpp"

class DropoutLayer : public Layer {
 public:
  DropoutLayer() {}
  explicit DropoutLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~DropoutLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Forward();
  void Release();
};

#endif  // SHADOW_LAYERS_DROPOUT_LAYER_HPP
