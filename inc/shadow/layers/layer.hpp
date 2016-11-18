#ifndef SHADOW_LAYERS_LAYER_HPP
#define SHADOW_LAYERS_LAYER_HPP

#include "shadow/blob.hpp"
#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

#include "shadow.pb.h"

class Layer {
 public:
  Layer() {}
  explicit Layer(const shadow::LayerParameter &layer_param)
      : layer_param_(layer_param),
        layer_name_(layer_param.name()),
        layer_type_(layer_param.type()) {}
  virtual ~Layer() {}

  virtual void Setup(VecBlob *blobs) {
    bottoms_.clear(), tops_.clear(), blobs_.clear();
    for (const auto &bottom_name : layer_param_.bottom()) {
      Blob<float> *bottom = get_blob_by_name(*blobs, bottom_name);
      if (bottom != nullptr) {
        if (bottom->num()) {
          bottoms_.push_back(bottom);
        } else {
          Fatal(layer_name_ + ": bottom blob(" + bottom_name +
                Util::format_vector(bottom->shape(), ",", "(", ")") +
                ") dimension mismatch!");
        }
      } else {
        Fatal(layer_name_ + ": bottom blob(" + bottom_name + ") not exist!");
      }
    }
    for (const auto &top_name : layer_param_.top()) {
      Blob<float> *top = get_blob_by_name(*blobs, top_name);
      if (top == nullptr) {
        top = new Blob<float>(top_name);
        blobs->push_back(top);
      }
      tops_.push_back(top);
    }
    for (const auto &proto_blob : layer_param_.blobs()) {
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

  inline const shadow::LayerParameter &param() const { return layer_param_; }
  inline shadow::LayerParameter &param() { return layer_param_; }

  inline const std::string &name() const { return layer_name_; }
  inline std::string &name() { return layer_name_; }

  inline const std::string &type() const { return layer_type_; }
  inline std::string &type() { return layer_type_; }

  inline int num_bottoms() const { return bottoms_.size(); }
  inline int num_tops() const { return tops_.size(); }
  inline int num_blobs() const { return blobs_.size(); }

  inline const VecBlob &bottoms() const { return bottoms_; }
  inline const VecBlob &tops() const { return tops_; }
  inline const VecBlob &blobs() const { return blobs_; }

  inline VecBlob &bottoms() { return bottoms_; }
  inline VecBlob &tops() { return tops_; }
  inline VecBlob &blobs() { return blobs_; }

  inline const Blob<float> *bottom(int i) const {
    if (i >= 0 && i < bottoms_.size()) {
      return bottoms_[i];
    } else {
      Fatal("Bottom " + Util::str(i) + " is not initialized!");
    }
    return nullptr;
  }
  inline const Blob<float> *top(int i) const {
    if (i >= 0 && i < tops_.size()) {
      return tops_[i];
    } else {
      Fatal("Top " + Util::str(i) + " is not initialized!");
    }
    return nullptr;
  }
  inline const Blob<float> *blob(int i) const {
    if (i >= 0 && i < blobs_.size()) {
      return blobs_[i];
    } else {
      Fatal("Blob " + Util::str(i) + " is not initialized!");
    }
    return nullptr;
  }

  inline Blob<float> *bottom(int i) {
    if (i >= 0 && i < bottoms_.size()) {
      return bottoms_[i];
    } else {
      Fatal("Bottom " + Util::str(i) + " is not initialized!");
    }
    return nullptr;
  }
  inline Blob<float> *top(int i) {
    if (i >= 0 && i < tops_.size()) {
      return tops_[i];
    } else {
      Fatal("Top " + Util::str(i) + " is not initialized!");
    }
    return nullptr;
  }
  inline Blob<float> *blob(int i) {
    if (i >= 0 && i < blobs_.size()) {
      return blobs_[i];
    } else {
      Fatal("Blob " + Util::str(i) + " is not initialized!");
    }
    return nullptr;
  }

  inline void set_blob(int i, const float *data) {
    if (i >= 0 && i < blobs_.size()) {
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
