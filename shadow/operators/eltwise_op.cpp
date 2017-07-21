#include "eltwise_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void EltwiseOp::Setup() {
  operation_ = get_single_argument<int>("operation", 1);
  const auto &coeff = get_repeated_argument<float>("coeff");
  coeff_size_ = static_cast<int>(coeff.size());

  CHECK_GE(bottoms_size(), 2);
  CHECK(coeff_size_ == 0 || coeff_size_ == bottoms_size())
      << "Eltwise op takes one coefficient per bottom blob.";
  CHECK(!(operation_ != 1 && coeff_size_))
      << "Eltwise op only takes coefficients for summation.";

  coeff_.resize(bottoms_size(), 1);
  for (int i = 0; i < coeff_size_; ++i) {
    coeff_[i] = coeff[i];
  }
}

void EltwiseOp::Reshape() {
  auto *top = mutable_tops<float>(0);

  for (int i = 1; i < bottoms_size(); ++i) {
    CHECK(bottoms<float>(i)->shape() == bottoms<float>(0)->shape());
  }
  top->reshape(bottoms<float>(0)->shape());

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms<float>(0)->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(top->shape(), ",", "(", ")");
}

void EltwiseOp::Forward() {
  auto *top = mutable_tops<float>(0);

  int count = bottoms<float>(0)->count();

  switch (operation_) {
    case 0:
      Blas::Mul(count, bottoms<float>(0)->data(), 0, bottoms<float>(1)->data(),
                0, top->mutable_data(), 0);
      for (int i = 2; i < bottoms_size(); ++i) {
        Blas::Mul(count, top->data(), 0, bottoms<float>(i)->data(), 0,
                  top->mutable_data(), 0);
      }
      break;
    case 1:
      Blas::Set(count, 0, top->mutable_data(), 0);
      for (int i = 0; i < bottoms_size(); ++i) {
        Blas::BlasSaxpy(count, coeff_[i], bottoms<float>(i)->data(), 0,
                        top->mutable_data(), 0);
      }
      break;
    case 2:
      Blas::Max(count, bottoms<float>(0)->data(), 0, bottoms<float>(1)->data(),
                0, top->mutable_data(), 0);
      for (int i = 2; i < bottoms_size(); ++i) {
        Blas::Max(count, top->data(), 0, bottoms<float>(i)->data(), 0,
                  top->mutable_data(), 0);
      }
      break;
    default:
      LOG(FATAL) << "Unknown elementwise operation.";
  }
}

void EltwiseOp::Release() {
  // DLOG(INFO) << "Free EltwiseOp!";
}

REGISTER_OPERATOR(Eltwise, EltwiseOp);

}  // namespace Shadow
