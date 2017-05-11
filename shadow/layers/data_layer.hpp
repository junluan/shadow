#ifndef SHADOW_LAYERS_DATA_LAYER_HPP
#define SHADOW_LAYERS_DATA_LAYER_HPP

#include "core/layer.hpp"

namespace Shadow {

class DataLayer : public Layer {
 public:
  DataLayer() {}
  explicit DataLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~DataLayer() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  float scale_;
  int num_mean_;

  BlobF mean_value_;
};

}  // namespace Shadow

#endif  // SHADOW_LAYERS_DATA_LAYER_HPP
