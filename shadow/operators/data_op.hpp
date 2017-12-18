#ifndef SHADOW_OPERATORS_DATA_OP_HPP
#define SHADOW_OPERATORS_DATA_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DataOp : public Operator {
 public:
  explicit DataOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    auto mean_value = get_repeated_argument<float>("mean_value");
    auto scale_value = get_repeated_argument<float>("scale_value");
    num_mean_ = 1, num_scale_ = 1;
    if (mean_value.size() > 1) {
      CHECK_EQ(mean_value.size(), bottoms<float>(0)->shape(1));
      num_mean_ = static_cast<int>(mean_value.size());
    } else if (mean_value.empty()) {
      mean_value.push_back(0);
    }
    mean_value_ =
        op_ws_->CreateBlob<float>({num_mean_}, op_name_ + "_mean_value");
    mean_value_->set_data(mean_value.data(), num_mean_);
    if (scale_value.size() > 1) {
      CHECK_EQ(scale_value.size(), bottoms<float>(0)->shape(1));
      num_scale_ = static_cast<int>(scale_value.size());
    } else if (scale_value.empty()) {
      scale_value.push_back(1);
    }
    scale_value_ =
        op_ws_->CreateBlob<float>({num_scale_}, op_name_ + "_scale_value");
    scale_value_->set_data(scale_value.data(), num_scale_);
  }

  void Reshape() override;
  void Forward() override;

 private:
  int num_mean_, num_scale_;

  BlobF *mean_value_ = nullptr, *scale_value_ = nullptr;
};

namespace Vision {

template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, int num_mean,
                   const T *mean_value, int num_scale, const T *scale_value,
                   T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DATA_OP_HPP
