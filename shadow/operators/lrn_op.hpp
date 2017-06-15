#ifndef SHADOW_OPERATORS_LRN_OP_HPP
#define SHADOW_OPERATORS_LRN_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class LRNOp : public Operator {
 public:
  LRNOp() {}
  explicit LRNOp(const shadow::OpParam &op_param) : Operator(op_param) {}
  ~LRNOp() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int size_, norm_region_;
  float alpha_, beta_, k_;

  BlobF scale_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_LRN_OP_HPP
