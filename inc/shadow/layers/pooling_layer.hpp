#ifndef SHADOW_LAYERS_POOLING_LAYER_HPP
#define SHADOW_LAYERS_POOLING_LAYER_HPP

#include "shadow/layers/layer.hpp"

class PoolingLayer : public Layer {
 public:
  PoolingLayer() {}
  explicit PoolingLayer(const shadow::LayerParameter &layer_param)
      : Layer(layer_param) {}
  ~PoolingLayer() { Release(); }

  void Setup(VecBlob *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int pool_type_, kernel_size_, stride_, pad_;
  bool global_pooling_;

#if defined(USE_CUDNN)
  cudnnPoolingDescriptor_t pooling_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
  cudnnPoolingMode_t mode_;
#endif
};

#endif  // SHADOW_LAYERS_POOLING_LAYER_HPP
