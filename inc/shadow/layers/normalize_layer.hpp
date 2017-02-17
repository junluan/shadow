#ifndef SHADOW_LAYERS_NORMALIZE_LAYER_HPP
#define SHADOW_LAYERS_NORMALIZE_LAYER_HPP

#include "shadow/layer.hpp"

class NormalizeLayer : public Layer {
 public:
  NormalizeLayer() {}
  explicit NormalizeLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~NormalizeLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  bool across_spatial_, channel_shared_;
  int spatial_dim_;
  float scale_;

  Blob<float> norm_, buffer_;
  Blob<float> sum_channel_multiplier_, sum_spatial_multiplier_;
};

#endif  // SHADOW_LAYERS_NORMALIZE_LAYER_HPP
