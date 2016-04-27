#ifndef SHADOW_CONNECTED_LAYER_H
#define SHADOW_CONNECTED_LAYER_H

#include "layer.hpp"

#include <string>

class ConnectedLayer : public Layer {
public:
  explicit ConnectedLayer(LayerType type);
  ~ConnectedLayer();

  void MakeConnectedLayer(SizeParams params, int outputs,
                          std::string activation);
  void ForwardLayer();
  float *GetOutData();

#ifdef USE_CUDA
  void CUDAForwardLayer();
#endif

#ifdef USE_CL
  void CLForwardLayer();
#endif

  void ReleaseLayer();

  Activation activation_;
  float *weights_, *biases_;

#ifdef USE_CUDA
  float *cuda_weights_, *cuda_biases_;
#endif

#ifdef USE_CL
  cl_mem cl_weights_, cl_biases_;
#endif
};

#endif // SHADOW_CONNECTED_LAYER_H
