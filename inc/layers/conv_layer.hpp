#ifndef SHADOW_CONV_LAYER_HPP
#define SHADOW_CONV_LAYER_HPP

#include "layer.hpp"

#include <string>

class ConvLayer : public Layer {
public:
  explicit ConvLayer(LayerType type);
  ~ConvLayer();

  void MakeConvLayer(SizeParams params, int out_num, int ksize, int stride,
                     int pad, std::string activation);
  void ForwardLayer();

#ifdef USE_CUDA
  void CUDAForwardLayer();
#endif

#ifdef USE_CL
  void CLForwardLayer();
#endif

  void ReleaseLayer();

  Activation activation_;
  int ksize_, stride_, pad_, out_map_size_, kernel_num_;
  float *filters_, *biases_, *col_image_;

#ifdef USE_CUDA
  float *cuda_filters_, *cuda_biases_, *cuda_col_image_;
#endif

#ifdef USE_CL
  cl_mem cl_filters_, cl_biases_, cl_col_image_;
#endif
};

#endif // SHADOW_CONV_LAYER_HPP
