#ifndef SHADOW_LAYERS_LAYER_HPP
#define SHADOW_LAYERS_LAYER_HPP

#include "shadow/blob.hpp"
#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

class Layer {
 public:
  Layer() {}
  explicit Layer(const shadow::LayerParameter &layer_param)
      : layer_param_(layer_param) {}

  virtual void Setup(VecBlob *blobs) {
    std::cout << "Setup Layer!" << std::endl;
  }
  virtual void Forward() { std::cout << "Forward Layer!" << std::endl; }
  virtual void Release() { std::cout << "Release Layer!" << std::endl; }

  shadow::LayerParameter layer_param_;

  VecBlob bottom_, top_;
};

typedef std::vector<Layer *> VecLayer;

#endif  // SHADOW_LAYERS_LAYER_HPP
