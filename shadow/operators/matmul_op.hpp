#ifndef SHADOW_OPERATORS_MATMUL_OP_HPP
#define SHADOW_OPERATORS_MATMUL_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class MatMulOp : public Operator {
 public:
  explicit MatMulOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    transpose_a_ = get_single_argument<bool>("transpose_a", false);
    transpose_b_ = get_single_argument<bool>("transpose_b", false);
  }

  void Forward() override;

 private:
  bool transpose_a_, transpose_b_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_MATMUL_OP_HPP
