#ifndef SHADOW_OPERATORS_CONCAT_OP_HPP
#define SHADOW_OPERATORS_CONCAT_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ConcatOp : public Operator {
 public:
  ConcatOp() {}
  explicit ConcatOp(const shadow::OpParam &op_param) : Operator(op_param) {}
  ~ConcatOp() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int concat_axis_, num_concats_, concat_input_size_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_CONCAT_OP_HPP
