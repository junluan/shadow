#ifndef SHADOW_LAYERS_ELTWISE_LAYER_HPP
#define SHADOW_LAYERS_ELTWISE_LAYER_HPP

#include "core/layer.hpp"

namespace Shadow {

class EltwiseLayer : public Layer {
 public:
  EltwiseLayer() {}
  explicit EltwiseLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~EltwiseLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int operation_, coeff_size_;
  VecFloat coeff_;
};

}  // namespace Shadow

#endif  // SHADOW_LAYERS_ELTWISE_LAYER_HPP
