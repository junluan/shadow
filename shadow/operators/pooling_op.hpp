#ifndef SHADOW_OPERATORS_POOLING_OP_HPP
#define SHADOW_OPERATORS_POOLING_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PoolingOp : public Operator {
 public:
  explicit PoolingOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~PoolingOp() { Release(); }

  void Setup();
  void Reshape();
  void Forward();
  void Release();

 private:
  int pool_type_, kernel_size_, stride_, pad_;
  bool global_pooling_;

#if defined(USE_CUDNN)
  cudnnPoolingDescriptor_t pooling_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
  cudnnPoolingMode_t mode_;
#endif
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_POOLING_OP_HPP
