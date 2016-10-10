#ifndef SHADOW_LAYERS_LAYER_HPP
#define SHADOW_LAYERS_LAYER_HPP

#include "shadow/blob.hpp"
#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

#include "shadow/proto/shadow.pb.h"

class Layer {
 public:
  Layer() {}
  explicit Layer(const shadow::LayerParameter &layer_param)
      : layer_param_(layer_param),
        layer_name_(layer_param.name()),
        layer_type_(layer_param.type()) {}

  virtual void Setup(VecBlob *blobs) { Info("Setup Layer!"); }
  virtual void Reshape() { Info("Reshape Layer!"); }
  virtual void Forward() { Info("Forward Layer!"); }
  virtual void Release() { Info("Release Layer!"); }

  shadow::LayerParameter layer_param_;
  std::string layer_name_;
  shadow::LayerType layer_type_;

  VecBlob bottom_, top_;
};

typedef std::vector<Layer *> VecLayer;

#endif  // SHADOW_LAYERS_LAYER_HPP
