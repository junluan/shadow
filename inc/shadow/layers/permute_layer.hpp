#ifndef SHADOW_LAYERS_PERMUTE_LAYER_HPP
#define SHADOW_LAYERS_PERMUTE_LAYER_HPP

#include "shadow/layers/layer.hpp"

class PermuteLayer : public Layer {
 public:
  PermuteLayer();
  explicit PermuteLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~PermuteLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Forward();
  void Release();

 private:
  Blob<int> *permute_order_, *old_steps_, *new_steps_;
};

#endif  // SHADOW_LAYERS_PERMUTE_LAYER_HPP
