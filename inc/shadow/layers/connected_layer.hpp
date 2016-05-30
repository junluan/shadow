#ifndef SHADOW_LAYERS_CONNECTED_LAYER_HPP
#define SHADOW_LAYERS_CONNECTED_LAYER_HPP

#include "shadow/layers/layer.hpp"

#include <string>

class ConnectedLayer : public Layer {
public:
  explicit ConnectedLayer(shadow::LayerParameter layer_param);
  ~ConnectedLayer();

  void MakeLayer(shadow::BlobShape *shape);
  void ForwardLayer();
  float *GetOutData();

#ifdef USE_CUDA
  void CUDAForwardLayer();
#endif

#ifdef USE_CL
  void CLForwardLayer();
#endif

  void ReleaseLayer();

  int num_output_;
  shadow::ActivateType activate_;
  float *weights_, *biases_;

#ifdef USE_CUDA
  float *cuda_weights_, *cuda_biases_;
#endif

#ifdef USE_CL
  cl_mem cl_weights_, cl_biases_;
#endif
};

#endif // SHADOW_LAYERS_CONNECTED_LAYER_HPP
