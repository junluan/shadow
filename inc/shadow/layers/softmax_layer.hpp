#ifndef SHADOW_LAYERS_SOFTMAX_LAYER_HPP
#define SHADOW_LAYERS_SOFTMAX_LAYER_HPP

#include "shadow/layers/layer.hpp"

class SoftmaxLayer : public Layer {
 public:
  SoftmaxLayer() {}
  explicit SoftmaxLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~SoftmaxLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int axis_, outer_num_, inner_num_;

  Blob<float> scale_;

#if defined(USE_CUDNN)
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
#endif
};

#endif  // SHADOW_LAYERS_SOFTMAX_LAYER_HPP
