#ifndef SHADOW_OPERATORS_DEFORM_PSROI_POOLING_OP_HPP
#define SHADOW_OPERATORS_DEFORM_PSROI_POOLING_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DeformPSROIPoolingOp : public Operator {
 public:
  explicit DeformPSROIPoolingOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    output_dim_ = get_single_argument<int>("output_dim", 0);
    group_size_ = get_single_argument<int>("group_size", 0);
    pooled_size_ = get_single_argument<int>("pooled_size", 0);
    part_size_ = get_single_argument<int>("part_size", 0);
    sample_per_part_ = get_single_argument<int>("sample_per_part", 1);
    CHECK_GT(output_dim_, 0) << "output_dim must be > 0";
    CHECK_GT(group_size_, 0) << "group_size must be > 0";
    CHECK_GT(pooled_size_, 0) << "pooled_size must be > 0";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);
    trans_std_ = get_single_argument<float>("trans_std", 0);
    no_trans_ = get_single_argument<bool>("no_trans", false);
    if (part_size_ == 0) {
      part_size_ = pooled_size_;
    }
  }

  void Forward() override;

 private:
  int output_dim_, group_size_, pooled_size_, part_size_, sample_per_part_;
  float spatial_scale_, trans_std_;
  bool no_trans_;
};

namespace Vision {

template <typename T>
void DeformPSROIPooling(const T *in_data, const VecInt &in_shape,
                        const T *roi_data, const T *trans_data,
                        const VecInt &trans_shape, int num_rois, int output_dim,
                        int group_size, int pooled_size, int part_size,
                        int sample_per_part, float spatial_scale,
                        float trans_std, bool no_trans, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DEFORM_PSROI_POOLING_OP_HPP
