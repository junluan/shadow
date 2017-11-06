#ifndef SHADOW_OPERATORS_BIAS_OP_HPP
#define SHADOW_OPERATORS_BIAS_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BiasOp : public Operator {
 public:
  explicit BiasOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);
    axis_ = bottoms<float>(0)->canonical_index(axis_);
    num_axis_ = get_single_argument<int>("num_axis", 1);
    CHECK_GE(num_axis_, -1);

    if (bottoms_size() == 1) {
      CHECK_EQ(blobs_size(), 1);
      bias_ = const_cast<BlobF *>(blobs<float>(0));
    } else {
      bias_ = const_cast<BlobF *>(bottoms<float>(1));
    }
  }

  void Reshape() override;
  void Forward() override;

 private:
  int axis_, num_axis_, bias_dim_, inner_dim_;

  BlobF *bias_ = nullptr;
};

namespace Vision {

template <typename T>
void Bias(const T *in_data, int count, const T *bias_data, int bias_dim,
          int inner_dim, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BIAS_OP_HPP
