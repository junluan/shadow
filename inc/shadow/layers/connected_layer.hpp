#ifndef SHADOW_LAYERS_CONNECTED_LAYER_HPP
#define SHADOW_LAYERS_CONNECTED_LAYER_HPP

#include "shadow/layers/layer.hpp"

class ConnectedLayer : public Layer {
public:
  explicit ConnectedLayer(shadow::LayerParameter layer_param);
  ~ConnectedLayer();

  void Setup(VecBlob *blobs);
  void Forward();
  void Release();

  void set_weights(float *weights) { weights_->set_data(weights); }
  void set_biases(float *biases) { biases_->set_data(biases); }

private:
  int num_output_;
  shadow::ActivateType activate_;

  Blob *weights_, *biases_;
};

#endif // SHADOW_LAYERS_CONNECTED_LAYER_HPP
