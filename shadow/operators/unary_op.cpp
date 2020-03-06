#include "unary_op.hpp"

namespace Shadow {

void UnaryOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  int count = bottom->count();

  switch (operation_) {
    case kAbs:
      return Blas::Abs(count, bottom->data(), 0, top->mutable_data(), 0,
                       op_ws_->Ctx());
    case kSquare:
      return Blas::Square(count, bottom->data(), 0, top->mutable_data(), 0,
                          op_ws_->Ctx());
    case kSqrt:
      return Blas::Sqrt(count, bottom->data(), 0, top->mutable_data(), 0,
                        op_ws_->Ctx());
    case kLog:
      return Blas::Log(count, bottom->data(), 0, top->mutable_data(), 0,
                       op_ws_->Ctx());
    case kExp:
      return Blas::Exp(count, bottom->data(), 0, top->mutable_data(), 0,
                       op_ws_->Ctx());
    case kSin:
      return Blas::Sin(count, bottom->data(), 0, top->mutable_data(), 0,
                       op_ws_->Ctx());
    case kCos:
      return Blas::Cos(count, bottom->data(), 0, top->mutable_data(), 0,
                       op_ws_->Ctx());
    case kTan:
      return Blas::Tan(count, bottom->data(), 0, top->mutable_data(), 0,
                       op_ws_->Ctx());
    case kAsin:
      return Blas::Asin(count, bottom->data(), 0, top->mutable_data(), 0,
                        op_ws_->Ctx());
    case kAcos:
      return Blas::Acos(count, bottom->data(), 0, top->mutable_data(), 0,
                        op_ws_->Ctx());
    case kAtan:
      return Blas::Atan(count, bottom->data(), 0, top->mutable_data(), 0,
                        op_ws_->Ctx());
    case kFloor:
      return Blas::Floor(count, bottom->data(), 0, top->mutable_data(), 0,
                         op_ws_->Ctx());
    case kCeil:
      return Blas::Ceil(count, bottom->data(), 0, top->mutable_data(), 0,
                        op_ws_->Ctx());
    case kNeg:
      return Blas::Neg(count, bottom->data(), 0, top->mutable_data(), 0,
                       op_ws_->Ctx());
    case kReciprocal:
      return Blas::Reciprocal(count, bottom->data(), 0, top->mutable_data(), 0,
                              op_ws_->Ctx());
    default:
      LOG(FATAL) << "Unknown unary operation " << operation_;
  }
}

REGISTER_OPERATOR(Unary, UnaryOp);

}  // namespace Shadow
