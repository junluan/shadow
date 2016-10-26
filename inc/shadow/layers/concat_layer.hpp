#ifndef SHADOW_LAYERS_CONCAT_LAYER_HPP
#define SHADOW_LAYERS_CONCAT_LAYER_HPP

#include "shadow/layers/layer.hpp"

class ConcatLayer : public Layer {
 public:
  ConcatLayer() {}
  explicit ConcatLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ConcatLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int concat_axis_, num_concats_, concat_input_size_;
};

#endif  // SHADOW_LAYERS_CONCAT_LAYER_HPP
