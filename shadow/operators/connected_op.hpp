#ifndef SHADOW_OPERATORS_CONNECTED_OP_HPP
#define SHADOW_OPERATORS_CONNECTED_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ConnectedOp : public Operator {
 public:
  ConnectedOp() {}
  explicit ConnectedOp(const shadow::OpParam &op_param) : Operator(op_param) {}
  ~ConnectedOp() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int num_output_;
  bool bias_term_, transpose_;

  BlobF biases_multiplier_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_CONNECTED_OP_HPP
