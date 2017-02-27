#ifndef SHADOW_LAYER_HPP
#define SHADOW_LAYER_HPP

#include "shadow/blob.hpp"
#include "shadow/util/log.hpp"
#include "shadow/util/util.hpp"

#if defined(USE_Protobuf)
#include "shadow.pb.h"

#else
#include "shadow/params.hpp"
#endif

class Layer {
 public:
  Layer() {}
  explicit Layer(const shadow::LayerParameter &layer_param)
      : layer_param_(layer_param),
        layer_name_(layer_param.name()),
        layer_type_(layer_param.type()) {}
  virtual ~Layer() {
    layer_param_.Clear();
    bottoms_.clear();
    tops_.clear();
    for (auto &blob : blobs_) {
      delete blob;
      blob = nullptr;
    }
    blobs_.clear();
  }

  virtual void Setup(VecBlobF *blobs) {
    bottoms_.clear(), tops_.clear(), blobs_.clear();
    for (const auto &bottom_name : layer_param_.bottom()) {
      BlobF *bottom = get_blob_by_name(*blobs, bottom_name);
      if (bottom != nullptr) {
        if (bottom->num()) {
          bottoms_.push_back(bottom);
        } else {
          LOG(FATAL) << layer_name_ << ": bottom blob(" << bottom_name
                     << Util::format_vector(bottom->shape(), ",", "(", ")")
                     << ") dimension mismatch!";
        }
      } else {
        LOG(FATAL) << layer_name_ << ": bottom blob(" << bottom_name
                   << ") not exist!";
      }
    }
    for (const auto &top_name : layer_param_.top()) {
      BlobF *top = get_blob_by_name(*blobs, top_name);
      if (top == nullptr) {
        top = new BlobF(top_name);
        blobs->push_back(top);
      }
      tops_.push_back(top);
    }
    for (const auto &proto_blob : layer_param_.blobs()) {
      BlobF *blob = new BlobF();
      const auto &dims = proto_blob.shape().dim();
      VecInt shape;
      int cc = 1, data_size = proto_blob.data_size();
      for (const auto &dim : dims) {
        cc *= dim;
        shape.push_back(dim);
      }
      if (data_size > 0) {
        CHECK_EQ(data_size, cc) << "Blob data size and blob shape are mismatch";
        blob->reshape(shape, true);
        blob->set_data(proto_blob.data().data(), true);
      } else {
        blob->reshape(shape);
      }
      blobs_.push_back(blob);
    }
  }
  virtual void Reshape() { LOG(INFO) << "Reshape Layer!"; }
  virtual void Forward() { LOG(INFO) << "Forward Layer!"; }
  virtual void Release() { LOG(INFO) << "Release Layer!"; }

  inline const shadow::LayerParameter &param() const { return layer_param_; }
  inline void set_param(const shadow::LayerParameter &param) {
    layer_param_ = param;
  }

  inline const std::string &name() const { return layer_name_; }
  inline void set_name(const std::string &name) { layer_name_ = name; }

  inline const std::string &type() const { return layer_type_; }
  inline void set_type(const std::string &type) { layer_type_ = type; }

  inline const VecBlobF &bottoms() const { return bottoms_; }
  inline VecBlobF *mutable_bottoms() { return &bottoms_; }
  inline const BlobF *bottoms(int i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, bottoms_.size());
    return bottoms_[i];
  }
  inline BlobF *mutable_bottoms(int i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, bottoms_.size());
    return bottoms_[i];
  }
  inline int bottoms_size() const { return bottoms_.size(); }

  inline const VecBlobF &tops() const { return tops_; }
  inline VecBlobF *mutable_tops() { return &tops_; }
  inline const BlobF *tops(int i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, tops_.size());
    return tops_[i];
  }
  inline BlobF *mutable_tops(int i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, tops_.size());
    return tops_[i];
  }
  inline int tops_size() const { return tops_.size(); }

  inline const VecBlobF &blobs() const { return blobs_; }
  inline VecBlobF *mutable_blobs() { return &blobs_; }
  inline const BlobF *blobs(int i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, blobs_.size());
    return blobs_[i];
  }
  inline BlobF *mutable_blobs(int i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, blobs_.size());
    return blobs_[i];
  }
  inline int blobs_size() const { return blobs_.size(); }
  inline void set_blob(int i, int count, const float *data) {
    CHECK_GE(i, 0);
    CHECK_LT(i, blobs_.size());
    CHECK_LE(count, blobs_[i]->count());
    blobs_[i]->set_data(data);
  }

 protected:
  shadow::LayerParameter layer_param_;

  std::string layer_name_, layer_type_;

  VecBlobF bottoms_, tops_, blobs_;
};

typedef std::vector<Layer *> VecLayer;

#endif  // SHADOW_LAYER_HPP
