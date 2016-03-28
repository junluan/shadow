#ifndef SHADOW_CONV_LAYER_H
#define SHADOW_CONV_LAYER_H

#include "layer.h"

#include <string>

class ConvLayer : public Layer {
public:
  ConvLayer(LayerType type);
  ~ConvLayer();

  void MakeConvLayer(SizeParams params, int out_num, int ksize, int stride,
                     int pad, std::string activation);
  void ForwardLayer();

#ifdef USE_CL
  void CLForwardLayer();
#endif

  void ReleaseLayer();

  Activation activation_;
  int ksize_, stride_, pad_, out_map_size_, kernel_num_;
  float *filters_, *biases_, *col_image_;

#ifdef USE_CL
  cl_mem cl_filters_, cl_biases_, cl_col_image_;
#endif
};

#endif // SHADOW_CONV_LAYER_H
