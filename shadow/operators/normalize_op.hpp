#ifndef SHADOW_OPERATORS_NORMALIZE_OP_HPP
#define SHADOW_OPERATORS_NORMALIZE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class NormalizeOp : public Operator {
 public:
  explicit NormalizeOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    across_spatial_ = get_single_argument<bool>("across_spatial", true);
    channel_shared_ = get_single_argument<bool>("channel_shared", true);

    CHECK_EQ(blobs_size(), 1);
    if (channel_shared_) {
      CHECK_EQ(blobs<float>(0)->count(), 1);
    } else {
      CHECK_EQ(blobs<float>(0)->count(), bottoms<float>(0)->shape(1));
    }
  }

  void Reshape() override;
  void Forward() override;

 private:
  bool across_spatial_, channel_shared_;
  int spatial_dim_;
  float scale_;

  BlobF *norm_ = nullptr, *buffer_ = nullptr;
  BlobF *sum_channel_multiplier_ = nullptr, *sum_spatial_multiplier_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_NORMALIZE_OP_HPP
