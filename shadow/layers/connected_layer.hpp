#ifndef SHADOW_LAYERS_CONNECTED_LAYER_HPP
#define SHADOW_LAYERS_CONNECTED_LAYER_HPP

#include "core/layer.hpp"

namespace Shadow {

class ConnectedLayer : public Layer {
 public:
  ConnectedLayer() {}
  explicit ConnectedLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ConnectedLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int num_output_;
  bool bias_term_, transpose_;

  BlobF biases_multiplier_;
};

}  // namespace Shadow

#endif  // SHADOW_LAYERS_CONNECTED_LAYER_HPP
