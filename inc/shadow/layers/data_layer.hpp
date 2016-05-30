#ifndef SHADOW_LAYERS_DATA_LAYER_HPP
#define SHADOW_LAYERS_DATA_LAYER_HPP

#include "shadow/layers/layer.hpp"

class DataLayer : public Layer {
public:
  explicit DataLayer(shadow::LayerParameter layer_param);
  ~DataLayer();

  void MakeLayer(shadow::BlobShape *shape);
  void ForwardLayer(float *in_data);

#ifdef USE_CUDA
  void CUDAForwardLayer(float *in_data);
#endif

#ifdef USE_CL
  void CLForwardLayer(float *in_data);
#endif

  void ReleaseLayer();

  float scale_, mean_value_;
};

#endif // SHADOW_LAYERS_DATA_LAYER_HPP
