#ifndef SHADOW_CORE_OPERATOR_HPP
#define SHADOW_CORE_OPERATOR_HPP

#include "blob.hpp"
#include "util/log.hpp"
#include "util/util.hpp"

#if defined(USE_Protobuf)
#include "proto/shadow.pb.h"

#else
#include "params.hpp"
#endif

namespace Shadow {

class Operator {
 public:
  Operator() {}
  explicit Operator(const shadow::OpParam &op_param) {
    op_param_ = op_param;
    op_name_ = op_param.name();
    op_type_ = op_param.type();
  }
  virtual ~Operator() {
    op_param_.Clear();
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
    for (const auto &bottom_name : op_param_.bottom()) {
      auto *bottom = get_blob_by_name(*blobs, bottom_name);
      if (bottom != nullptr) {
        if (bottom->num()) {
          bottoms_.push_back(bottom);
        } else {
          LOG(FATAL) << op_name_ << ": bottom blob(" << bottom_name
                     << Util::format_vector(bottom->shape(), ",", "(", ")")
                     << ") dimension mismatch!";
        }
      } else {
        LOG(FATAL) << op_name_ << ": bottom blob(" << bottom_name
                   << ") not exist!";
      }
    }
    for (const auto &top_name : op_param_.top()) {
      auto *top = get_blob_by_name(*blobs, top_name);
      if (top == nullptr) {
        top = new BlobF(top_name);
        blobs->push_back(top);
      }
      tops_.push_back(top);
    }
    for (const auto &proto_blob : op_param_.blobs()) {
      const auto &dims = proto_blob.shape().dim();
      VecInt shape;
      int cc = 1, data_size = proto_blob.data_size();
      for (const auto dim : dims) {
        cc *= dim;
        shape.push_back(dim);
      }
      auto *blob = new BlobF(shape, "params", true);
      if (data_size > 0) {
        CHECK_EQ(data_size, cc) << "Blob data size and blob shape are mismatch";
        blob->set_data(proto_blob.data().data(), data_size);
      }
      blobs_.push_back(blob);
    }
  }
  virtual void Reshape() { LOG(INFO) << "Reshape Operator!"; }
  virtual void Forward() { LOG(INFO) << "Forward Operator!"; }
  virtual void Release() { LOG(INFO) << "Release Operator!"; }

  const shadow::OpParam &param() const { return op_param_; }
  void set_param(const shadow::OpParam &param) { op_param_ = param; }

  const std::string &name() const { return op_name_; }
  void set_name(const std::string &name) { op_name_ = name; }

  const std::string &type() const { return op_type_; }
  void set_type(const std::string &type) { op_type_ = type; }

  const VecBlobF &bottoms() const { return bottoms_; }
  VecBlobF *mutable_bottoms() { return &bottoms_; }
  const BlobF *bottoms(int i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, bottoms_.size());
    return bottoms_[i];
  }
  BlobF *mutable_bottoms(int i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, bottoms_.size());
    return bottoms_[i];
  }
  int bottoms_size() const { return bottoms_.size(); }

  const VecBlobF &tops() const { return tops_; }
  VecBlobF *mutable_tops() { return &tops_; }
  const BlobF *tops(int i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, tops_.size());
    return tops_[i];
  }
  BlobF *mutable_tops(int i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, tops_.size());
    return tops_[i];
  }
  int tops_size() const { return tops_.size(); }

  const VecBlobF &blobs() const { return blobs_; }
  VecBlobF *mutable_blobs() { return &blobs_; }
  const BlobF *blobs(int i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, blobs_.size());
    return blobs_[i];
  }
  BlobF *mutable_blobs(int i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, blobs_.size());
    return blobs_[i];
  }
  int blobs_size() const { return blobs_.size(); }
  void set_blob(int i, int count, const float *data) {
    CHECK_GE(i, 0);
    CHECK_LT(i, blobs_.size());
    blobs_[i]->set_data(data, count);
  }

 protected:
  shadow::OpParam op_param_;

  std::string op_name_, op_type_;

  VecBlobF bottoms_, tops_, blobs_;
};

typedef std::vector<Operator *> VecOp;

}  // namespace Shadow

#endif  // SHADOW_CORE_OPERATOR_HPP
