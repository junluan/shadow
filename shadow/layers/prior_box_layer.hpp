#ifndef SHADOW_LAYERS_PRIOR_BOX_LAYER_HPP
#define SHADOW_LAYERS_PRIOR_BOX_LAYER_HPP

#include "core/layer.hpp"

namespace Shadow {

class PriorBoxLayer : public Layer {
 public:
  PriorBoxLayer() {}
  explicit PriorBoxLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~PriorBoxLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int num_priors_;
  float step_, offset_;
  bool flip_, clip_, is_initial_;
  VecFloat min_sizes_, max_sizes_, aspect_ratios_, variance_, top_data_;
};

}  // namespace Shadow

#endif  // SHADOW_LAYERS_PRIOR_BOX_LAYER_HPP
