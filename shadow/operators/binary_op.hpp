#ifndef SHADOW_OPERATORS_BINARY_OP_HPP
#define SHADOW_OPERATORS_BINARY_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BinaryOp : public Operator {
 public:
  explicit BinaryOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
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

  void Reshape() override;
  void Forward() override;

 private:
  enum { kAdd = 0, kSub = 1, kMul = 2, kDiv = 3, kPow = 4, kMax = 5, kMin = 6 };

  int operation_;

  BlobF *scalar_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BINARY_OP_HPP
