#ifndef SHADOW_DATA_LAYER_HPP
#define SHADOW_DATA_LAYER_HPP

#include "layer.hpp"

class DataLayer : public Layer {
public:
  explicit DataLayer(LayerType type);
  ~DataLayer();

  void MakeDataLayer(SizeParams params);
  void ForwardLayer(float *in_data);

#ifdef USE_CUDA
  void CUDAForwardLayer(float *in_data);
#endif

#ifdef USE_CL
  void CLForwardLayer(float *in_data);
#endif

  void ReleaseLayer();

  float scale_;
  float mean_value_;
};

#endif // SHADOW_DATA_LAYER_HPP
