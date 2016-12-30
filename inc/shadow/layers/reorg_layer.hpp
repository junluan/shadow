#ifndef SHADOW_LAYERS_REORG_LAYER_HPP
#define SHADOW_LAYERS_REORG_LAYER_HPP

#include "shadow/layers/layer.hpp"

class ReorgLayer : public Layer {
 public:
  ReorgLayer() {}
  explicit ReorgLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ReorgLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int stride_;
};

#endif  // SHADOW_LAYERS_REORG_LAYER_HPP
