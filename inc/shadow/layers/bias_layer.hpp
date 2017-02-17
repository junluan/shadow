#ifndef SHADOW_LAYERS_BIAS_LAYER_HPP
#define SHADOW_LAYERS_BIAS_LAYER_HPP

#include "shadow/layer.hpp"

class BiasLayer : public Layer {
 public:
  BiasLayer() {}
  explicit BiasLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~BiasLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int axis_, num_axis_, bias_dim_, inner_dim_;

  Blob<float> *bias_;
};

#endif  // SHADOW_LAYERS_BIAS_LAYER_HPP
