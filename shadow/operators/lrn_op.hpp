#ifndef SHADOW_OPERATORS_LRN_OP_HPP
#define SHADOW_OPERATORS_LRN_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class LRNOp : public Operator {
 public:
  explicit LRNOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    size_ = get_single_argument<int>("local_size", 5);
    CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
    alpha_ = get_single_argument<float>("alpha", 1);
    beta_ = get_single_argument<float>("beta", 0.75);
    norm_region_ = get_single_argument<int>("norm_region", 0);
    CHECK_EQ(norm_region_, 0)
        << "Currently only support norm region method: Across Channels!";
    k_ = get_single_argument<float>("k", 1);
  }

  void Forward() override;

 private:
  int size_, norm_region_;
  float alpha_, beta_, k_;

  BlobF *scale_ = nullptr;
};

namespace Vision {

template <typename T>
void LRN(const T *in_data, const VecInt &in_shape, int size, float alpha,
         float beta, float k, T *scale_data, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_LRN_OP_HPP
