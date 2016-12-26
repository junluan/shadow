#ifndef SHADOW_LAYERS_BATCH_NORM_LAYER_HPP
#define SHADOW_LAYERS_BATCH_NORM_LAYER_HPP

#include "shadow/layers/layer.hpp"

class BatchNormLayer : public Layer {
 public:
  BatchNormLayer() {}
  explicit BatchNormLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~BatchNormLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  bool use_global_stats_;
  float scale_;
  int channels_, spatial_dim_;

  Blob<float> mean_, variance_, temp_;
  Blob<float> sum_batch_multiplier_, sum_spatial_multiplier_, batch_by_channel_;
};

#endif  // SHADOW_LAYERS_BATCH_NORM_LAYER_HPP
