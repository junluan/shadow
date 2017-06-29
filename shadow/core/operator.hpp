#ifndef SHADOW_CORE_OPERATOR_HPP
#define SHADOW_CORE_OPERATOR_HPP

#include "blob.hpp"
#include "factory.hpp"
#include "params.hpp"
#include "workspace.hpp"

#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

class Operator {
 public:
  explicit Operator(const shadow::OpParam &op_param, Workspace *ws);
  virtual ~Operator();

  virtual void Setup() { LOG(INFO) << "Setup Operator!"; }
  virtual void Reshape() { LOG(INFO) << "Reshape Operator!"; }
  virtual void Forward() { LOG(INFO) << "Forward Operator!"; }
  virtual void Release() { LOG(INFO) << "Release Operator!"; }

  bool HasArgument(const std::string &name) const {
    return arg_helper_.HasArgument(name);
  }
  template <typename T>
  T GetSingleArgument(const std::string &name, const T &default_value) const {
    return arg_helper_.template GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  bool HasSingleArgumentOfType(const std::string &name) const {
    return arg_helper_.template HasSingleArgumentOfType<T>(name);
  }
  template <typename T>
  std::vector<T> GetRepeatedArgument(
      const std::string &name, const std::vector<T> &default_value = {}) const {
    return arg_helper_.template GetRepeatedArgument<T>(name, default_value);
  }

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
  ArgumentHelper arg_helper_;
  Workspace *op_ws_ = nullptr;

  std::string op_name_, op_type_;

  VecBlobF bottoms_, tops_, blobs_;
};

typedef std::vector<Operator *> VecOp;

}  // namespace Shadow

#endif  // SHADOW_CORE_OPERATOR_HPP
