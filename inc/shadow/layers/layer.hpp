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

  virtual void Setup(VecBlob *blobs) {
    for (int i = 0; i < layer_param_.bottom_size(); ++i) {
      Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(i));
      if (bottom != nullptr) {
        if (bottom->num()) {
          bottom_.push_back(bottom);
        } else {
          Fatal(layer_name_ + ": bottom blob(" + layer_param_.bottom(i) +
                Util::format_vector(bottom->shape(), ",", "(", ")") +
                ") dimension mismatch!");
        }
      } else {
        Fatal(layer_name_ + ": bottom blob(" + layer_param_.bottom(0) +
              ") not exist!");
      }
    }
    for (int i = 0; i < layer_param_.top_size(); ++i) {
      Blob<float> *top = find_blob_by_name(*blobs, layer_param_.top(i));
      if (top == nullptr) {
        top = new Blob<float>(layer_param_.top(i));
        blobs->push_back(top);
      }
      top_.push_back(top);
    }
  }
  virtual void Reshape() { Info("Reshape Layer!"); }
  virtual void Forward() { Info("Forward Layer!"); }
  virtual void Release() { Info("Release Layer!"); }

  virtual inline const std::string name() { return layer_name_; }
  virtual inline const shadow::LayerType type() { return layer_type_; }
  virtual inline Blob<float> *bottom(int i) { return bottom_[i]; }
  virtual inline Blob<float> *top(int i) { return top_[i]; }

 protected:
  shadow::LayerParameter layer_param_;
  std::string layer_name_;
  shadow::LayerType layer_type_;

  VecBlob bottom_, top_;
};

typedef std::vector<Layer *> VecLayer;

#endif  // SHADOW_LAYERS_LAYER_HPP
