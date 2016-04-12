#ifndef SHADOW_DATA_LAYER_H
#define SHADOW_DATA_LAYER_H

#include "layer.h"

class DataLayer : public Layer {
public:
  DataLayer(LayerType type);
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

#endif // SHADOW_DATA_LAYER_H
