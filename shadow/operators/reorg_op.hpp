#ifndef SHADOW_OPERATORS_REORG_OP_HPP
#define SHADOW_OPERATORS_REORG_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReorgOp : public Operator {
 public:
  ReorgOp() {}
  explicit ReorgOp(const shadow::OpParam &op_param) : Operator(op_param) {}
  ~ReorgOp() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int stride_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_REORG_OP_HPP
