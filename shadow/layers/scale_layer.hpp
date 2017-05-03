#ifndef SHADOW_LAYERS_SCALE_LAYER_HPP
#define SHADOW_LAYERS_SCALE_LAYER_HPP

#include "core/layer.hpp"

class ScaleLayer : public Layer {
 public:
  ScaleLayer() {}
  explicit ScaleLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ScaleLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  bool bias_term_;
  int axis_, num_axis_, scale_dim_, inner_dim_, bias_param_id_;

  BlobF *scale_ = nullptr, *bias_ = nullptr;
};

#endif  // SHADOW_LAYERS_SCALE_LAYER_HPP
