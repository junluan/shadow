#ifndef SHADOW_LAYERS_BIAS_LAYER_HPP
#define SHADOW_LAYERS_BIAS_LAYER_HPP

#include "core/layer.hpp"

namespace Shadow {

class BiasLayer : public Layer {
 public:
  BiasLayer() {}
  explicit BiasLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~BiasLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int axis_, num_axis_, bias_dim_, inner_dim_;

  BlobF *bias_;
};

}  // namespace Shadow

#endif  // SHADOW_LAYERS_BIAS_LAYER_HPP
