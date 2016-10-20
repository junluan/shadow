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
      Blob<float> *bottom = get_blob_by_name(*blobs, layer_param_.bottom(i));
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
      Blob<float> *top = get_blob_by_name(*blobs, layer_param_.top(i));
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

  virtual inline const shadow::LayerParameter param() const {
    return layer_param_;
  }
  virtual inline shadow::LayerParameter &param() { return layer_param_; }

  virtual inline const std::string name() const { return layer_name_; }
  virtual inline const std::string type() const { return layer_type_; }

  virtual inline int num_bottoms() { return bottom_.size(); }
  virtual inline int num_tops() { return top_.size(); }

  virtual inline const Blob<float> *bottom(int i) const { return bottom_[i]; }
  virtual inline const Blob<float> *top(int i) const { return top_[i]; }

  virtual inline Blob<float> *bottom(int i) { return bottom_[i]; }
  virtual inline Blob<float> *top(int i) { return top_[i]; }

 protected:
  shadow::LayerParameter layer_param_;

  std::string layer_name_, layer_type_;

  VecBlob bottom_, top_;
};

typedef std::vector<Layer *> VecLayer;

#endif  // SHADOW_LAYERS_LAYER_HPP
