#ifndef SHADOW_OPERATORS_BATCH_NORM_OP_HPP
#define SHADOW_OPERATORS_BATCH_NORM_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BatchNormOp : public Operator {
 public:
  explicit BatchNormOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    use_global_stats_ = get_single_argument<bool>("use_global_stats", true);
    eps_ = get_single_argument<float>("eps", 1e-5);
    if (bottoms<float>(0)->num_axes() == 1) {
      channels_ = 1;
    } else {
      channels_ = bottoms<float>(0)->shape(1);
    }

    if (use_global_stats_) {
      CHECK_EQ(blobs_size(), 3);
      CHECK_EQ(blobs<float>(2)->count(), 1);
    }
  }

  void Reshape() override;
  void Forward() override;

 private:
  bool use_global_stats_;
  float eps_, scale_;
  int channels_, spatial_dim_;

  BlobF *mean_ = nullptr, *variance_ = nullptr, *temp_ = nullptr;
  BlobF *sum_batch_multiplier_ = nullptr, *sum_spatial_multiplier_ = nullptr,
        *batch_by_channel_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BATCH_NORM_OP_HPP
