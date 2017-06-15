#include "eltwise_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void EltwiseOp::Setup(VecBlobF *blobs) {
  Operator::Setup(blobs);

  const auto &eltwise_param = op_param_.eltwise_param();

  operation_ = eltwise_param.operation();
  coeff_size_ = eltwise_param.coeff_size();

  CHECK_GE(bottoms_.size(), 2);
  CHECK(coeff_size_ == 0 || coeff_size_ == bottoms_.size())
      << "Eltwise op takes one coefficient per bottom blob.";
  CHECK(!(operation_ != shadow::EltwiseParam_EltwiseOp_Sum && coeff_size_))
      << "Eltwise op only takes coefficients for summation.";

  coeff_.resize(bottoms_.size(), 1);
  for (int i = 0; i < coeff_size_; ++i) {
    coeff_[i] = eltwise_param.coeff(i);
  }
}

void EltwiseOp::Reshape() {
  for (int i = 1; i < bottoms_.size(); ++i) {
    CHECK(bottoms_[i]->shape() == bottoms_[0]->shape());
  }
  tops_[0]->reshape(bottoms_[0]->shape());

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void EltwiseOp::Forward() {
  int count = bottoms_[0]->count();

  switch (operation_) {
    case shadow::EltwiseParam_EltwiseOp_Prod:
      Blas::Mul(count, bottoms_[0]->data(), 0, bottoms_[1]->data(), 0,
                tops_[0]->mutable_data(), 0);
      for (int i = 2; i < bottoms_.size(); ++i) {
        Blas::Mul(count, tops_[0]->data(), 0, bottoms_[i]->data(), 0,
                  tops_[0]->mutable_data(), 0);
      }
      break;
    case shadow::EltwiseParam_EltwiseOp_Sum:
      Blas::Set(count, 0, tops_[0]->mutable_data(), 0);
      for (int i = 0; i < bottoms_.size(); ++i) {
        Blas::BlasSaxpy(count, coeff_[i], bottoms_[i]->data(), 0,
                        tops_[0]->mutable_data(), 0);
      }
      break;
    case shadow::EltwiseParam_EltwiseOp_Max:
      Blas::Max(count, bottoms_[0]->data(), 0, bottoms_[1]->data(), 0,
                tops_[0]->mutable_data(), 0);
      for (int i = 2; i < bottoms_.size(); ++i) {
        Blas::Max(count, tops_[0]->data(), 0, bottoms_[i]->data(), 0,
                  tops_[0]->mutable_data(), 0);
      }
      break;
    default:
      LOG(FATAL) << "Unknown elementwise operation.";
  }
}

void EltwiseOp::Release() {
  // DLOG(INFO) << "Free EltwiseOp!";
}

}  // namespace Shadow
