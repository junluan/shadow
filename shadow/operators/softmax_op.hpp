#ifndef SHADOW_OPERATORS_SOFTMAX_OP_HPP
#define SHADOW_OPERATORS_SOFTMAX_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class SoftmaxOp : public Operator {
 public:
  SoftmaxOp() {}
  explicit SoftmaxOp(const shadow::OpParam &op_param) : Operator(op_param) {}
  ~SoftmaxOp() { Release(); }

  void Setup(VecBlobF *blobs);
  void Reshape();
  void Forward();
  void Release();

 private:
  int axis_, outer_num_, inner_num_;

  BlobF scale_;

#if defined(USE_CUDNN)
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
#endif
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SOFTMAX_OP_HPP