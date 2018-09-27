#include "eltwise_op.hpp"

namespace Shadow {

void EltwiseOp::Forward() {
  const auto *bottom_0 = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_GE(bottoms_size(), 2);
  for (int n = 1; n < bottoms_size(); ++n) {
    CHECK(bottoms<float>(n)->shape() == bottom_0->shape());
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
      Blas::Mul(count, bottoms<float>(0)->data(), 0, bottoms<float>(1)->data(),
                0, top->mutable_data(), 0);
      for (int n = 2; n < bottoms_size(); ++n) {
        Blas::Mul(count, top->data(), 0, bottoms<float>(n)->data(), 0,
                  top->mutable_data(), 0);
      }
      break;
    case kSum:
      Blas::Set(count, 0, top->mutable_data(), 0);
      for (int n = 0; n < bottoms_size(); ++n) {
        Blas::BlasSaxpy(count, coeff[n], bottoms<float>(n)->data(), 0,
                        top->mutable_data(), 0, op_ws_->BlasHandle());
      }
      break;
    case kMax:
      Blas::Max(count, bottoms<float>(0)->data(), 0, bottoms<float>(1)->data(),
                0, top->mutable_data(), 0);
      for (int n = 2; n < bottoms_size(); ++n) {
        Blas::Max(count, top->data(), 0, bottoms<float>(n)->data(), 0,
                  top->mutable_data(), 0);
      }
      break;
    case kMin:
      Blas::Min(count, bottoms<float>(0)->data(), 0, bottoms<float>(1)->data(),
                0, top->mutable_data(), 0);
      for (int n = 2; n < bottoms_size(); ++n) {
        Blas::Min(count, top->data(), 0, bottoms<float>(n)->data(), 0,
                  top->mutable_data(), 0);
      }
      break;
    default:
      LOG(FATAL) << "Unknown elementwise operation " << operation_;
  }
}

REGISTER_OPERATOR(Eltwise, EltwiseOp);

}  // namespace Shadow
