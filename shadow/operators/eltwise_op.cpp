#include "eltwise_op.hpp"

namespace Shadow {

void EltwiseOp::Forward() {
  const auto bottom_0 = bottoms(0);
  auto top = tops(0);

  CHECK_GE(bottoms_size(), 2);
  for (int n = 1; n < bottoms_size(); ++n) {
    CHECK(bottoms(n)->shape() == bottom_0->shape());
  }
  top->reshape(bottom_0->shape());

  int coeff_size = static_cast<int>(coeff_.size());

  CHECK(coeff_size == 0 || coeff_size == bottoms_size())
      << "Eltwise op takes one coefficient per bottom blob.";
  CHECK(!(operation_ != 1 && coeff_size))
      << "Eltwise op only takes coefficients for summation.";

  VecFloat coeff(bottoms_size(), 1);
  for (int n = 0; n < coeff_size; ++n) {
    coeff[n] = coeff_[n];
  }

  int count = bottom_0->count();

  // Prod: 0, Sum: 1, Max: 2, Min: 3
  switch (operation_) {
    case kProd:
      Blas::Mul(count, bottoms(0)->data<float>(), 0, bottoms(1)->data<float>(),
                0, top->mutable_data<float>(), 0, ws_->Ctx());
      for (int n = 2; n < bottoms_size(); ++n) {
        Blas::Mul(count, top->data<float>(), 0, bottoms(n)->data<float>(), 0,
                  top->mutable_data<float>(), 0, ws_->Ctx());
      }
      break;
    case kSum:
      Blas::Set(count, 0, top->mutable_data<float>(), 0, ws_->Ctx());
      for (int n = 0; n < bottoms_size(); ++n) {
        Blas::BlasSaxpy(count, coeff[n], bottoms(n)->data<float>(), 0,
                        top->mutable_data<float>(), 0, ws_->Ctx());
      }
      break;
    case kMax:
      Blas::Max(count, bottoms(0)->data<float>(), 0, bottoms(1)->data<float>(),
                0, top->mutable_data<float>(), 0, ws_->Ctx());
      for (int n = 2; n < bottoms_size(); ++n) {
        Blas::Max(count, top->data<float>(), 0, bottoms(n)->data<float>(), 0,
                  top->mutable_data<float>(), 0, ws_->Ctx());
      }
      break;
    case kMin:
      Blas::Min(count, bottoms(0)->data<float>(), 0, bottoms(1)->data<float>(),
                0, top->mutable_data<float>(), 0, ws_->Ctx());
      for (int n = 2; n < bottoms_size(); ++n) {
        Blas::Min(count, top->data<float>(), 0, bottoms(n)->data<float>(), 0,
                  top->mutable_data<float>(), 0, ws_->Ctx());
      }
      break;
    default:
      LOG(FATAL) << "Unknown elementwise operation " << operation_;
  }
}

REGISTER_OPERATOR(Eltwise, EltwiseOp);

}  // namespace Shadow
