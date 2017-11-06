#ifndef SHADOW_OPERATORS_PSROI_POOLING_OP_HPP
#define SHADOW_OPERATORS_PSROI_POOLING_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PSROIPoolingOp : public Operator {
 public:
  explicit PSROIPoolingOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    output_dim_ = get_single_argument<int>("output_dim", 0);
    group_size_ = get_single_argument<int>("group_size", 0);
    CHECK_GT(output_dim_, 0) << "output_dim must be > 0";
    CHECK_GT(group_size_, 0) << "group_size must be > 0";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);
    CHECK_EQ(bottoms_size(), 2);
    pooled_h_ = group_size_, pooled_w_ = group_size_;
  }

  void Reshape() override;
  void Forward() override;

 private:
  int output_dim_, group_size_, pooled_h_, pooled_w_;
  float spatial_scale_;
};

namespace Vision {

template <typename T>
void PSROIPooling(const T *in_data, const VecInt &in_shape, const T *roi_data,
                  int num_rois, int output_dim, int group_size, int pooled_h,
                  int pooled_w, float spatial_scale, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PSROI_POOLING_OP_HPP
