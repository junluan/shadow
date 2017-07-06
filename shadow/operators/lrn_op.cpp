#include "lrn_op.hpp"
#include "core/image.hpp"

namespace Shadow {

void LRNOp::Setup() {
  size_ = arg_helper_.GetSingleArgument<int>("local_size", 5);
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  alpha_ = arg_helper_.GetSingleArgument<float>("alpha", 1);
  beta_ = arg_helper_.GetSingleArgument<float>("beta", 0.75);
  norm_region_ = arg_helper_.GetSingleArgument<int>("norm_region", 0);
  k_ = arg_helper_.GetSingleArgument<float>("k", 1);
}

void LRNOp::Reshape() {
  tops_[0]->reshape(bottoms_[0]->shape());

  scale_.reshape(bottoms_[0]->shape());

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void LRNOp::Forward() {
  Image::LRN(bottoms_[0]->data(), bottoms_[0]->shape(), size_, alpha_, beta_,
             k_, scale_.mutable_data(), tops_[0]->mutable_data());
}

void LRNOp::Release() {
  scale_.clear();

  // DLOG(INFO) << "Free LRNOp!";
}

REGISTER_OPERATOR(LRN, LRNOp);

}  // namespace Shadow
