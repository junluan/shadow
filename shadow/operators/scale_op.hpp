#ifndef SHADOW_OPERATORS_SCALE_OP_HPP
#define SHADOW_OPERATORS_SCALE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ScaleOp : public Operator {
 public:
  ScaleOp() {}
  explicit ScaleOp(const shadow::OpParam &op_param) : Operator(op_param) {}
  ~ScaleOp() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  bool bias_term_;
  int axis_, num_axis_, scale_dim_, inner_dim_, bias_param_id_;

  BlobF *scale_ = nullptr, *bias_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SCALE_OP_HPP
