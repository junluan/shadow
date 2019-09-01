#ifndef SHADOW_OPERATORS_SOFTMAX_OP_HPP
#define SHADOW_OPERATORS_SOFTMAX_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class SoftmaxOp : public Operator {
 public:
  SoftmaxOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);

#if defined(USE_CUDNN)
    cudnn::createTensorDesc<float>(&bottom_top_desc_);
#endif
  }
  ~SoftmaxOp() override {
#if defined(USE_CUDNN)
    if (bottom_top_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bottom_top_desc_);
      bottom_top_desc_ = nullptr;
    }
#endif
  }

  void Forward() override;

 private:
  int axis_;

#if defined(USE_CUDNN)
  cudnnTensorDescriptor_t bottom_top_desc_ = nullptr;
#endif
};

namespace Vision {

template <typename T>
void Softmax(const T *in_data, int outer_num, int channels, int inner_num,
             T *val_data, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SOFTMAX_OP_HPP
