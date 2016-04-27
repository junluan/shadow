#ifndef SHADOW_NETWORK_H
#define SHADOW_NETWORK_H

#include "connected_layer.hpp"
#include "conv_layer.hpp"
#include "data_layer.hpp"
#include "dropout_layer.hpp"
#include "layer.hpp"
#include "pooling_layer.hpp"

#include <vector>

class Network {
public:
  int batch_;
  int in_c_, in_h_, in_w_;
  int in_num_, out_num_;

  int class_num_, grid_size_, sqrt_box_, box_num_;

  int num_layers_;
  std::vector<Layer *> layers_;

  void MakeNetwork(int n);
  float *PredictNetwork(float *in_data);
  int GetNetworkOutputSize();
  float *GetNetworkOutput();
  void SetNetworkBatch(int batch);
  void ReleaseNetwork();

private:
  void ForwardNetwork(float *in_data);

#ifdef USE_CUDA
  void CUDAForwardNetwork(float *in_data);
#endif

#ifdef USE_CL
  void CLForwardNetwork(float *in_data);
#endif
};

#endif // SHADOW_NETWORK_H
