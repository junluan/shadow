#ifndef SHADOW_LAYERS_CONV_LAYER_HPP
#define SHADOW_LAYERS_CONV_LAYER_HPP

#include "shadow/layers/layer.hpp"

#include <string>

class ConvLayer : public Layer {
public:
  explicit ConvLayer(shadow::LayerParameter layer_param);
  ~ConvLayer();

  void MakeLayer(shadow::BlobShape *shape);
  void ForwardLayer();

#ifdef USE_CUDA
  void CUDAForwardLayer();
#endif

#ifdef USE_CL
  void CLForwardLayer();
#endif

  void ReleaseLayer();

  int num_output_, kernel_size_, stride_, pad_, out_map_size_, kernel_num_;
  shadow::ActivateType activate_;
  float *filters_, *biases_, *col_image_;

#ifdef USE_CUDA
  float *cuda_filters_, *cuda_biases_, *cuda_col_image_;
#endif

#ifdef USE_CL
  cl_mem cl_filters_, cl_biases_, cl_col_image_;
#endif
};

#endif // SHADOW_LAYERS_CONV_LAYER_HPP
