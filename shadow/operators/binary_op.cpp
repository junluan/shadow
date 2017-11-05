#include "binary_op.hpp"

namespace Shadow {

void BinaryOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK(bottom->shape() == scalar_->shape());

  if (bottom != top && scalar_ != top) {
    top->reshape(bottom->shape());
  }

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << ", "
             << scalar_->name()
             << Util::format_vector(scalar_->shape(), ",", "(", ")") << " -> "
             << operation_ << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void BinaryOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int count = top->count();

  switch (operation_) {
    case kAdd:
      return Blas::Add(count, bottom->data(), 0, scalar_->data(), 0,
                       top->mutable_data(), 0);
    case kSub:
      return Blas::Sub(count, bottom->data(), 0, scalar_->data(), 0,
                       top->mutable_data(), 0);
    case kMul:
      return Blas::Mul(count, bottom->data(), 0, scalar_->data(), 0,
                       top->mutable_data(), 0);
    case kDiv:
      return Blas::Div(count, bottom->data(), 0, scalar_->data(), 0,
                       top->mutable_data(), 0);
    case kPow:
      return Blas::Pow(count, bottom->data(), 0, scalar_->data(), 0,
                       top->mutable_data(), 0);
    case kMax:
      return Blas::Max(count, bottom->data(), 0, scalar_->data(), 0,
                       top->mutable_data(), 0);
    case kMin:
      return Blas::Min(count, bottom->data(), 0, scalar_->data(), 0,
                       top->mutable_data(), 0);
    default:
      LOG(FATAL) << "Unknown binary operation " << operation_;
  }
}

REGISTER_OPERATOR(Binary, BinaryOp);

}  // namespace Shadow