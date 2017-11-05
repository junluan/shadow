#include "eltwise_op.hpp"

namespace Shadow {

void EltwiseOp::Reshape() {
  auto *top = mutable_tops<float>(0);

  for (int i = 1; i < bottoms_size(); ++i) {
    CHECK(bottoms<float>(i)->shape() == bottoms<float>(0)->shape());
  }
  top->reshape(bottoms<float>(0)->shape());

  VecString str;
  for (int i = 0; i < bottoms_size(); ++i) {
    const auto *bottom = bottoms<float>(i);
    str.push_back(
        Util::format_vector(bottom->shape(), ",", bottom->name() + "(", ")"));
  }
  DLOG(INFO) << op_name_ << "(" << op_type_
             << "): " << Util::format_vector(str, " + ") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void EltwiseOp::Forward() {
  auto *top = mutable_tops<float>(0);

  int count = bottoms<float>(0)->count();

  // Prod: 0, Sum: 1, Max: 2
  switch (operation_) {
    case kProd:
      Blas::Mul(count, bottoms<float>(0)->data(), 0, bottoms<float>(1)->data(),
                0, top->mutable_data(), 0);
      for (int i = 2; i < bottoms_size(); ++i) {
        Blas::Mul(count, top->data(), 0, bottoms<float>(i)->data(), 0,
                  top->mutable_data(), 0);
      }
      break;
    case kSum:
      Blas::Set(count, 0, top->mutable_data(), 0);
      for (int i = 0; i < bottoms_size(); ++i) {
        Blas::BlasSaxpy(count, coeff_[i], bottoms<float>(i)->data(), 0,
                        top->mutable_data(), 0);
      }
      break;
    case kMax:
      Blas::Max(count, bottoms<float>(0)->data(), 0, bottoms<float>(1)->data(),
                0, top->mutable_data(), 0);
      for (int i = 2; i < bottoms_size(); ++i) {
        Blas::Max(count, top->data(), 0, bottoms<float>(i)->data(), 0,
                  top->mutable_data(), 0);
      }
      break;
    default:
      LOG(FATAL) << "Unknown elementwise operation " << operation_;
  }
}

REGISTER_OPERATOR(Eltwise, EltwiseOp);

}  // namespace Shadow
