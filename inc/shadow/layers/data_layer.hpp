#ifndef SHADOW_LAYERS_DATA_LAYER_HPP
#define SHADOW_LAYERS_DATA_LAYER_HPP

#include "shadow/layers/layer.hpp"

class DataLayer : public Layer {
public:
  explicit DataLayer(shadow::LayerParameter layer_param);
  ~DataLayer();

  void Setup(VecBlob *blobs);
  void Forward();
  void Release();

private:
  float scale_, mean_value_;
};

#endif // SHADOW_LAYERS_DATA_LAYER_HPP
