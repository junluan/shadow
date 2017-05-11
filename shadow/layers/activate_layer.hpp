#ifndef SHADOW_LAYERS_ACTIVATE_LAYER_HPP
#define SHADOW_LAYERS_ACTIVATE_LAYER_HPP

#include "core/layer.hpp"

namespace Shadow {

class ActivateLayer : public Layer {
 public:
  ActivateLayer() {}
  explicit ActivateLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ActivateLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int activate_type_;
  bool channel_shared_;
};

}  // namespace Shadow

#endif  // SHADOW_ACTIVATE_LAYER_HPP
