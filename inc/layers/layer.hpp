#ifndef SHADOW_LAYER_HPP
#define SHADOW_LAYER_HPP

#include "activations.hpp"
#include "kernel.hpp"

#include <iostream>
#include <string>

struct SizeParams {
  int batch;
  int in_num;
  int in_c, in_h, in_w;
};

enum LayerType { kData, kConvolutional, kMaxpool, kConnected, kDropout, kCost };

class Layer {
public:
  std::string layer_name_;
  LayerType layer_type_;

  int batch_;
  int in_c_, in_h_, in_w_;
  int out_c_, out_h_, out_w_;

  int in_num_, out_num_;
  float *in_data_, *out_data_;

#ifdef USE_CUDA
  float *cuda_in_data_, *cuda_out_data_;
#endif

#ifdef USE_CL
  cl_mem cl_in_data_, cl_out_data_;
#endif

  virtual void ForwardLayer() { std::cout << "Forward Layer!" << std::endl; }
  virtual void ForwardLayer(float *in_data) {}
  virtual float *GetOutData() { return NULL; }

#ifdef USE_CUDA
  virtual void CUDAForwardLayer() {
    std::cout << "CUDAForward Layer!" << std::endl;
  }
  virtual void CUDAForwardLayer(float *in_data) {
    std::cout << "CUDAForward Layer!" << std::endl;
  }
#endif

#ifdef USE_CL
  virtual void CLForwardLayer() {
    std::cout << "CLForward Layer!" << std::endl;
  }
  virtual void CLForwardLayer(float *in_data) {
    std::cout << "CLForward Layer!" << std::endl;
  }
#endif

  virtual void ReleaseLayer() { std::cout << "Free Layer!" << std::endl; }
};

#endif // SHADOW_LAYER_HPP
