#ifndef SHADOW_LAYERS_LAYER_HPP
#define SHADOW_LAYERS_LAYER_HPP

#include "shadow/blob.hpp"
#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

class Layer {
public:
  virtual void MakeLayer(Blob<BType> *blob) {
    std::cout << "Make Layer!" << std::endl;
  }
  virtual void ForwardLayer() { std::cout << "Forward Layer!" << std::endl; }
  virtual void ForwardLayer(float *in_data) {
    std::cout << "Forward Layer!" << std::endl;
  }

  virtual void ReleaseLayer() { std::cout << "Free Layer!" << std::endl; }

  shadow::LayerParameter layer_param_;

  Blob<BType> *in_blob_, *out_blob_;
};

#endif // SHADOW_LAYERS_LAYER_HPP
