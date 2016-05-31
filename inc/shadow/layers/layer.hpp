#ifndef SHADOW_LAYERS_LAYER_HPP
#define SHADOW_LAYERS_LAYER_HPP

#include "shadow/blob.hpp"
#include "shadow/kernel.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/util.hpp"

#include <iostream>
#include <string>

class Layer {
public:
  float *in_data_, *out_data_;

  shadow::LayerParameter layer_param_;
  Blob *in_blob, *out_blob;

#ifdef USE_CUDA
  float *cuda_in_data_, *cuda_out_data_;
#endif

#ifdef USE_CL
  cl_mem cl_in_data_, cl_out_data_;
#endif

  virtual void MakeLayer(Blob *blob) {
    std::cout << "Make Layer!" << std::endl;
  }
  virtual void ForwardLayer() { std::cout << "Forward Layer!" << std::endl; }
  virtual void ForwardLayer(float *in_data) {
    std::cout << "Forward Layer!" << std::endl;
  }
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

#endif // SHADOW_LAYERS_LAYER_HPP
