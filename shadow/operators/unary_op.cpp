#include "unary_op.hpp"

namespace Shadow {

void UnaryOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void UnaryOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int count = bottom->count();

  switch (operation_) {
    case kAbs:
      return Blas::Abs(count, bottom->data(), 0, top->mutable_data(), 0);
    case kSquare:
      return Blas::Square(count, bottom->data(), 0, top->mutable_data(), 0);
    case kSqrt:
      return Blas::Sqrt(count, bottom->data(), 0, top->mutable_data(), 0);
    case kLog:
      return Blas::Log(count, bottom->data(), 0, top->mutable_data(), 0);
    case kExp:
      return Blas::Exp(count, bottom->data(), 0, top->mutable_data(), 0);
    case kSin:
      return Blas::Sin(count, bottom->data(), 0, top->mutable_data(), 0);
    case kCos:
      return Blas::Cos(count, bottom->data(), 0, top->mutable_data(), 0);
    case kTan:
      return Blas::Tan(count, bottom->data(), 0, top->mutable_data(), 0);
    case kAsin:
      return Blas::Asin(count, bottom->data(), 0, top->mutable_data(), 0);
    case kAcos:
      return Blas::Acos(count, bottom->data(), 0, top->mutable_data(), 0);
    case kAtan:
      return Blas::Atan(count, bottom->data(), 0, top->mutable_data(), 0);
    case kFloor:
      return Blas::Floor(count, bottom->data(), 0, top->mutable_data(), 0);
    case kCeil:
      return Blas::Ceil(count, bottom->data(), 0, top->mutable_data(), 0);
    default:
      LOG(FATAL) << "Unknown unary operation " << operation_;
  }
}

REGISTER_OPERATOR(Unary, UnaryOp);

}  // namespace Shadow