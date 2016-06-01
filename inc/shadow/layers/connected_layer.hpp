#ifndef SHADOW_LAYERS_CONNECTED_LAYER_HPP
#define SHADOW_LAYERS_CONNECTED_LAYER_HPP

#include "shadow/layers/layer.hpp"

class ConnectedLayer : public Layer {
public:
  explicit ConnectedLayer(shadow::LayerParameter layer_param);
  ~ConnectedLayer();

  void MakeLayer(Blob<BType> *blob);

  void ForwardLayer();

  void ReleaseLayer();

  int num_output_;
  shadow::ActivateType activate_;

  float *out_data_;

  Blob<BType> *weights_, *biases_;
};

#endif // SHADOW_LAYERS_CONNECTED_LAYER_HPP
