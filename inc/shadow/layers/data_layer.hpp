#ifndef SHADOW_LAYERS_DATA_LAYER_HPP
#define SHADOW_LAYERS_DATA_LAYER_HPP

#include "shadow/layers/layer.hpp"

class DataLayer : public Layer {
 public:
  DataLayer() {}
  explicit DataLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~DataLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  float scale_, mean_value_;
};

#endif  // SHADOW_LAYERS_DATA_LAYER_HPP
