#ifndef SHADOW_LAYERS_CONNECTED_LAYER_HPP
#define SHADOW_LAYERS_CONNECTED_LAYER_HPP

#include "shadow/layers/layer.hpp"

class ConnectedLayer : public Layer {
 public:
  ConnectedLayer() {}
  explicit ConnectedLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ConnectedLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

  void set_weights(const float *weights) { weights_->set_data(weights); }
  void set_biases(const float *biases) { biases_->set_data(biases); }

 private:
  int num_output_;
  shadow::ActivateType activate_;

  Blob<float> *weights_, *biases_;
};

#endif  // SHADOW_LAYERS_CONNECTED_LAYER_HPP
