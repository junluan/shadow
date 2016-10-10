#ifndef SHADOW_LAYERS_FLATTEN_LAYER_HPP
#define SHADOW_LAYERS_FLATTEN_LAYER_HPP

#include "shadow/layers/layer.hpp"

class FlattenLayer : public Layer {
 public:
  FlattenLayer();
  explicit FlattenLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~FlattenLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Forward();
  void Release();

 private:
  int start_axis_, end_axis_;
};

#endif  // SHADOW_LAYERS_FLATTEN_LAYER_HPP
