#include "activate_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

void ActivateOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> t"
             << activate_type_ << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void ActivateOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0);
  }

  // PRelu: 0, Relu: 1, Leaky: 2, Sigmoid: 3, SoftPlus: 4, Tanh: 5
  if (activate_type_ == kRelu || activate_type_ == kLeaky ||
      activate_type_ == kSigmoid || activate_type_ == kSoftPlus ||
      activate_type_ == kTanh) {
    Vision::Activate(top->mutable_data(), top->count(), activate_type_, slope_);
  } else if (activate_type_ == kPRelu) {
    Vision::PRelu(top->mutable_data(), top->shape(), channel_shared_,
                  blobs<float>(0)->data());
  }
}

REGISTER_OPERATOR(Activate, ActivateOp);

}  // namespace Shadow
