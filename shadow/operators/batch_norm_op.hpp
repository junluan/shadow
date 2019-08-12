#ifndef SHADOW_OPERATORS_BATCH_NORM_OP_HPP
#define SHADOW_OPERATORS_BATCH_NORM_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BatchNormOp : public Operator {
 public:
  BatchNormOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    use_global_stats_ = get_single_argument<bool>("use_global_stats", true);
    eps_ = get_single_argument<float>("eps", 1e-5);

#if defined(USE_CUDNN)
    use_cudnn_ = use_global_stats_;
    if (use_cudnn_) {
      cudnn::createTensorDesc<float>(&bottom_top_desc_);
      cudnn::createTensorDesc<float>(&param_desc_);
    }
#endif
  }
  ~BatchNormOp() {
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
  bool use_global_stats_, use_cudnn_ = false;

#if defined(USE_CUDNN)
  cudnnTensorDescriptor_t bottom_top_desc_ = nullptr, param_desc_ = nullptr;
#endif
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BATCH_NORM_OP_HPP
