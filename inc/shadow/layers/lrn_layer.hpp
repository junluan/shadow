#ifndef SHADOW_LAYERS_LRN_LAYER_HPP
#define SHADOW_LAYERS_LRN_LAYER_HPP

#include "shadow/layers/layer.hpp"

class LRNLayer : public Layer {
 public:
  LRNLayer() {}
  explicit LRNLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~LRNLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int size_;
  float alpha_, beta_, k_;

  shadow::LRNParameter_NormRegion norm_region_;

  Blob<float> scale_;
};

#endif  // SHADOW_LAYERS_LRN_LAYER_HPP
