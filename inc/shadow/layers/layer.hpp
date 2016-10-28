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
    bottoms_.clear(), tops_.clear(), blobs_.clear();
    for (int i = 0; i < layer_param_.bottom_size(); ++i) {
      Blob<float> *bottom = get_blob_by_name(*blobs, layer_param_.bottom(i));
      if (bottom != nullptr) {
        if (bottom->num()) {
          bottoms_.push_back(bottom);
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
      tops_.push_back(top);
    }
    for (int i = 0; i < layer_param_.blobs_size(); ++i) {
      const shadow::Blob &proto_blob = layer_param_.blobs(i);
      int data_size = proto_blob.data_size(), count = proto_blob.count();
      Blob<float> *blob;
      if (data_size > 0) {
        blob = new Blob<float>(data_size, proto_blob.data().data());
      } else {
        if (count > 0) {
          blob = new Blob<float>(count);
        } else {
          blob = new Blob<float>();
        }
      }
      blobs_.push_back(blob);
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

  virtual inline int num_bottoms() const { return bottoms_.size(); }
  virtual inline int num_tops() const { return tops_.size(); }
  virtual inline int num_blobs() const { return blobs_.size(); }

  virtual inline Blob<float> *bottom(int i) const {
    if (i < bottoms_.size()) {
      return bottoms_[i];
    }
    return nullptr;
  }
  virtual inline Blob<float> *top(int i) const {
    if (i < tops_.size()) {
      return tops_[i];
    }
    return nullptr;
  }
  virtual inline Blob<float> *blob(int i) const {
    if (i < blobs_.size()) {
      return blobs_[i];
    }
    return nullptr;
  }

  virtual inline void set_blob(int i, const float *data) {
    if (i < blobs_.size()) {
      blobs_[i]->set_data(data);
    } else {
      Fatal("Blob " + Util::str(i) + " is not initialized!");
    }
  }

 protected:
  shadow::LayerParameter layer_param_;

  std::string layer_name_, layer_type_;

  VecBlob bottoms_, tops_, blobs_;
};

typedef std::vector<Layer *> VecLayer;

#endif  // SHADOW_LAYERS_LAYER_HPP
