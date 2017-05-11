#ifndef SHADOW_LAYERS_NORMALIZE_LAYER_HPP
#define SHADOW_LAYERS_NORMALIZE_LAYER_HPP

#include "core/layer.hpp"

namespace Shadow {

class NormalizeLayer : public Layer {
 public:
  NormalizeLayer() {}
  explicit NormalizeLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~NormalizeLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  bool across_spatial_, channel_shared_;
  int spatial_dim_;
  float scale_;

  BlobF norm_, buffer_;
  BlobF sum_channel_multiplier_, sum_spatial_multiplier_;
};

}  // namespace Shadow

#endif  // SHADOW_LAYERS_NORMALIZE_LAYER_HPP
