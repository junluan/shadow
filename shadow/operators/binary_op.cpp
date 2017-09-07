#include "binary_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void BinaryOp::Setup() {
  operation_ = get_single_argument<int>("operation", -1);
  CHECK_GE(operation_, 0);
  CHECK_LE(operation_, 6);
  if (has_argument("scalar")) {
    auto scalar_data = get_single_argument<float>("scalar", 0);
    scalar_ = op_ws_->CreateBlob<float>(bottoms<float>(0)->shape(),
                                        op_name_ + "_param_scalar");
    Blas::Set(scalar_->count(), scalar_data, scalar_->mutable_data(), 0);
  } else if (bottoms_size() > 1) {
    scalar_ = const_cast<BlobF *>(bottoms<float>(1));
  } else if (blobs_size() > 0) {
    scalar_ = const_cast<BlobF *>(blobs<float>(0));
  } else {
    LOG(FATAL) << "Missing right blob for doing binary operation";
  }
}

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

void BinaryOp::Release() {
  // DLOG(INFO) << "Free BinaryOp!";
}

REGISTER_OPERATOR(Binary, BinaryOp);

}  // namespace Shadow