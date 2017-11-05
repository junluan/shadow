#ifndef SHADOW_OPERATORS_ELTWISE_OP_HPP
#define SHADOW_OPERATORS_ELTWISE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class EltwiseOp : public Operator {
 public:
  explicit EltwiseOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
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

  void Reshape() override;
  void Forward() override;

 private:
  enum { kProd = 0, kSum = 1, kMax = 2 };

  int operation_, coeff_size_;
  VecFloat coeff_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ELTWISE_OP_HPP
