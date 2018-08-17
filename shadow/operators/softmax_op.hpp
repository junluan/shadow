#ifndef SHADOW_OPERATORS_SOFTMAX_OP_HPP
#define SHADOW_OPERATORS_SOFTMAX_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class SoftmaxOp : public Operator {
 public:
  explicit SoftmaxOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);

#if defined(USE_CUDNN)
    cudnn::createTensorDesc<float>(&bottom_desc_);
    cudnn::createTensorDesc<float>(&top_desc_);
#endif
  }
  ~SoftmaxOp() override {
#if defined(USE_CUDNN)
    if (bottom_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bottom_desc_);
      bottom_desc_ = nullptr;
    }
    if (top_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(top_desc_);
      top_desc_ = nullptr;
    }
#endif
  }

  void Forward() override;

 private:
  int axis_, outer_num_, inner_num_;

  BlobF *scale_ = nullptr;

#if defined(USE_CUDNN)
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
#endif
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SOFTMAX_OP_HPP
