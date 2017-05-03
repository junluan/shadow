#ifndef SHADOW_LAYERS_PERMUTE_LAYER_HPP
#define SHADOW_LAYERS_PERMUTE_LAYER_HPP

#include "core/layer.hpp"

class PermuteLayer : public Layer {
 public:
  PermuteLayer() {}
  explicit PermuteLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~PermuteLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int num_axes_;
  VecInt permute_order_data_;

  BlobI permute_order_, old_steps_, new_steps_;
};

#endif  // SHADOW_LAYERS_PERMUTE_LAYER_HPP
