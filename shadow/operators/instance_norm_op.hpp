#ifndef SHADOW_OPERATORS_INSTANCE_NORM_OP_HPP
#define SHADOW_OPERATORS_INSTANCE_NORM_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class InstanceNormOp : public Operator {
 public:
  InstanceNormOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    eps_ = get_single_argument<float>("eps", 1e-5);

#if defined(USE_CUDNN)
    cudnn::createTensorDesc<float>(&bottom_top_desc_);
    cudnn::createTensorDesc<float>(&param_desc_);
#endif
  }
  ~InstanceNormOp() {
#if defined(USE_CUDNN)
    if (bottom_top_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bottom_top_desc_);
      bottom_top_desc_ = nullptr;
    }
    if (param_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(param_desc_);
      param_desc_ = nullptr;
    }
#endif
  }

  void Forward() override;

 private:
  float eps_;

#if defined(USE_CUDNN)
  cudnnTensorDescriptor_t bottom_top_desc_ = nullptr, param_desc_ = nullptr;
#endif
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_INSTANCE_NORM_OP_HPP
