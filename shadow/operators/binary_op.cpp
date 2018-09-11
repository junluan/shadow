#include "binary_op.hpp"

namespace Shadow {

void BinaryOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (!has_scalar_arg_) {
    CHECK_EQ(bottoms_size(), 2);
    scalar_ = const_cast<BlobF *>(bottoms<float>(1));
  }

  if (bottom != top && scalar_ != top) {
    top->reshape(bottom->shape());
  }

  int count = top->count();

  switch (operation_) {
    case kAdd:
      if (has_scalar_arg_) {
        return Blas::Add(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Add(count, bottom->data(), 0, scalar_->data(), 0,
                         top->mutable_data(), 0);
      }
    case kSub:
      if (has_scalar_arg_) {
        return Blas::Sub(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Sub(count, bottom->data(), 0, scalar_->data(), 0,
                         top->mutable_data(), 0);
      }
    case kMul:
      if (has_scalar_arg_) {
        return Blas::Mul(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Mul(count, bottom->data(), 0, scalar_->data(), 0,
                         top->mutable_data(), 0);
      }
    case kDiv:
      if (has_scalar_arg_) {
        return Blas::Div(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Div(count, bottom->data(), 0, scalar_->data(), 0,
                         top->mutable_data(), 0);
      }
    case kPow:
      if (has_scalar_arg_) {
        return Blas::Pow(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Pow(count, bottom->data(), 0, scalar_->data(), 0,
                         top->mutable_data(), 0);
      }
    case kMax:
      if (has_scalar_arg_) {
        return Blas::Max(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Max(count, bottom->data(), 0, scalar_->data(), 0,
                         top->mutable_data(), 0);
      }
    case kMin:
      if (has_scalar_arg_) {
        return Blas::Min(count, bottom->data(), 0, scalar_data_,
                         top->mutable_data(), 0);
      } else {
        return Blas::Min(count, bottom->data(), 0, scalar_->data(), 0,
                         top->mutable_data(), 0);
      }
    default:
      LOG(FATAL) << "Unknown binary operation " << operation_;
  }
}

REGISTER_OPERATOR(Binary, BinaryOp);

}  // namespace Shadow
