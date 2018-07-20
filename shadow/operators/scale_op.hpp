#ifndef SHADOW_OPERATORS_SCALE_OP_HPP
#define SHADOW_OPERATORS_SCALE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ScaleOp : public Operator {
 public:
  explicit ScaleOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);
    auto has_scale = get_single_argument<bool>("has_scale", true);
    auto has_bias = get_single_argument<bool>("has_bias", true);
    CHECK(has_scale || has_bias);
    auto scale_value = get_repeated_argument<float>("scale_value");
    auto bias_value = get_repeated_argument<float>("bias_value");

    if (scale_value.empty() && bias_value.empty()) {
      CHECK_GE(blobs_size(), 1);
      if (has_scale && has_bias) {
        CHECK_EQ(blobs_size(), 2);
        scale_ = const_cast<BlobF *>(blobs<float>(0));
        bias_ = const_cast<BlobF *>(blobs<float>(1));
      } else if (has_scale) {
        scale_ = const_cast<BlobF *>(blobs<float>(0));
        bias_ = op_ws_->CreateBlob<float>(scale_->shape(),
                                          op_name_ + "_bias_value");
        Blas::Set(bias_->count(), 0, bias_->mutable_data(), 0);
      } else {
        bias_ = const_cast<BlobF *>(blobs<float>(0));
        scale_ = op_ws_->CreateBlob<float>(bias_->shape(),
                                           op_name_ + "_scale_value");
        Blas::Set(scale_->count(), 1, scale_->mutable_data(), 0);
      }
    } else {
      axis_ = 1;
      int dim = bottoms<float>(0)->shape(axis_);
      if (scale_value.size() > 1) {
        CHECK_EQ(scale_value.size(), dim);
      } else if (scale_value.size() == 1) {
        scale_value = VecFloat(dim, scale_value[0]);
      } else {
        scale_value = VecFloat(dim, 1);
      }
      if (bias_value.size() > 1) {
        CHECK_EQ(bias_value.size(), dim);
      } else if (bias_value.size() == 1) {
        bias_value = VecFloat(dim, bias_value[0]);
      } else {
        bias_value = VecFloat(dim, 0);
      }
      scale_ = op_ws_->CreateBlob<float>({dim}, op_name_ + "_scale_value");
      bias_ = op_ws_->CreateBlob<float>({dim}, op_name_ + "_bias_value");
      scale_->set_data(scale_value.data(), dim);
      bias_->set_data(bias_value.data(), dim);
    }
  }

  void Reshape() override;
  void Forward() override;

 private:
  int axis_, scale_dim_, inner_dim_;

  BlobF *scale_ = nullptr, *bias_ = nullptr;
};

namespace Vision {

template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SCALE_OP_HPP
