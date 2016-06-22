#ifndef SHADOW_LAYERS_CONV_LAYER_HPP
#define SHADOW_LAYERS_CONV_LAYER_HPP

#include "shadow/layers/layer.hpp"

class ConvLayer : public Layer {
 public:
  ConvLayer() {}
  explicit ConvLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~ConvLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Forward();
  void Release();

  int kernel_size() { return kernel_size_; }

  void set_filters(float *filters) { filters_->set_data(filters); }
  void set_biases(float *biases) { biases_->set_data(biases); }

 private:
  int num_output_, kernel_size_, stride_, pad_, out_map_size_, kernel_num_;
  shadow::ActivateType activate_;

  Blob<float> *filters_, *biases_, *col_image_;
};

#endif  // SHADOW_LAYERS_CONV_LAYER_HPP
