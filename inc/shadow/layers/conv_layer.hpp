#ifndef SHADOW_LAYERS_CONV_LAYER_HPP
#define SHADOW_LAYERS_CONV_LAYER_HPP

#include "shadow/layers/layer.hpp"

class ConvLayer : public Layer {
public:
  explicit ConvLayer(shadow::LayerParameter layer_param);
  ~ConvLayer();

  void MakeLayer(Blob<BType> *blob);

  void ForwardLayer();

  void ReleaseLayer();

  int num_output_, kernel_size_, stride_, pad_, out_map_size_, kernel_num_;
  shadow::ActivateType activate_;

  Blob<BType> *filters_, *biases_, *col_image_;
};

#endif // SHADOW_LAYERS_CONV_LAYER_HPP
