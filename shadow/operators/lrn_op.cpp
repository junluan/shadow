#include "lrn_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

void LRNOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  top->reshape(bottom->shape());

  scale_ = op_ws_->CreateBlob<float>(op_name_ + "_scale");
  scale_->reshape(bottom->shape());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void LRNOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::LRN(bottom->data(), bottom->shape(), size_, alpha_, beta_, k_,
              scale_->mutable_data(), top->mutable_data());
}

REGISTER_OPERATOR(LRN, LRNOp);

}  // namespace Shadow
