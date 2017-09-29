#include "lrn_op.hpp"
#include "core/image.hpp"

namespace Shadow {

void LRNOp::Setup() {
  size_ = get_single_argument<int>("local_size", 5);
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  alpha_ = get_single_argument<float>("alpha", 1);
  beta_ = get_single_argument<float>("beta", 0.75);
  norm_region_ = get_single_argument<int>("norm_region", 0);
  CHECK_EQ(norm_region_, 0)
      << "Currently only support norm region method: Across Channels!";
  k_ = get_single_argument<float>("k", 1);
}

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

  Image::LRN(bottom->data(), bottom->shape(), size_, alpha_, beta_, k_,
             scale_->mutable_data(), top->mutable_data());
}

void LRNOp::Release() {
  // DLOG(INFO) << "Free LRNOp!";
}

REGISTER_OPERATOR(LRN, LRNOp);

}  // namespace Shadow
