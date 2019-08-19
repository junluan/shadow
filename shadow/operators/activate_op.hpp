#ifndef SHADOW_OPERATORS_ACTIVATE_OP_HPP
#define SHADOW_OPERATORS_ACTIVATE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ActivateOp : public Operator {
 public:
  ActivateOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    activate_type_ = get_single_argument<int>("type", 1);
    slope_ = get_single_argument<float>("slope", 0.1);
    CHECK_GE(activate_type_, 0);
    CHECK_LE(activate_type_, 6);

#if defined(USE_CUDNN)
    use_cudnn_ = activate_type_ == kRelu || activate_type_ == kSigmoid ||
                 activate_type_ == kTanh;
    if (use_cudnn_) {
      cudnn::createActivationDesc<float>(&activate_desc_);
      cudnn::createTensorDesc<float>(&bottom_top_desc_);
    }
#endif
  }
  ~ActivateOp() override {
#if defined(USE_CUDNN)
    if (activate_desc_ != nullptr) {
      cudnnDestroyActivationDescriptor(activate_desc_);
      activate_desc_ = nullptr;
    }
    if (bottom_top_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bottom_top_desc_);
      bottom_top_desc_ = nullptr;
    }
#endif
  }

  void Forward() override;

  enum {
    kPRelu = 0,
    kRelu = 1,
    kLeaky = 2,
    kSigmoid = 3,
    kSoftPlus = 4,
    kTanh = 5,
    kRelu6 = 6
  };

 private:
  int activate_type_;
  float slope_;
  bool use_cudnn_ = false;

#if defined(USE_CUDNN)
  cudnnActivationDescriptor_t activate_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_top_desc_ = nullptr;
#endif
};

namespace Vision {

template <typename T>
void Activate(const T *in_data, T *out_data, int count, int type,
              float slope = 0.1);

template <typename T>
void PRelu(const T *in_data, T *out_data, const VecInt &in_shape,
           bool channel_shared, const T *slope_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ACTIVATE_OP_HPP
