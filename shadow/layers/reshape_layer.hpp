#ifndef SHADOW_LAYERS_RESHAPE_LAYER_HPP
#define SHADOW_LAYERS_RESHAPE_LAYER_HPP

#include "core/layer.hpp"

class ReshapeLayer : public Layer {
 public:
  ReshapeLayer() {}
  explicit ReshapeLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ReshapeLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int axis_, num_axes_, inferred_axis_, constant_count_;
  VecInt copy_axes_;
};

#endif  // SHADOW_LAYERS_RESHAPE_LAYER_HPP
