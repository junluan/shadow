#ifndef SHADOW_OPERATORS_BIAS_OP_HPP
#define SHADOW_OPERATORS_BIAS_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BiasOp : public Operator {
 public:
  BiasOp() {}
  explicit BiasOp(const shadow::OpParam &op_param) : Operator(op_param) {}
  ~BiasOp() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int axis_, num_axis_, bias_dim_, inner_dim_;

  BlobF *bias_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BIAS_OP_HPP
