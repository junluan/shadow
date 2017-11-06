#ifndef SHADOW_OPERATORS_DATA_OP_HPP
#define SHADOW_OPERATORS_DATA_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DataOp : public Operator {
 public:
  explicit DataOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    scale_ = get_single_argument<float>("scale", 1);
    auto mean_value = get_repeated_argument<float>("mean_value");
    num_mean_ = 1;
    if (mean_value.size() > 1) {
      CHECK_EQ(mean_value.size(), bottoms<float>(0)->shape(1));
      num_mean_ = static_cast<int>(mean_value.size());
    } else if (mean_value.empty()) {
      mean_value.push_back(0);
    }
    mean_value_ =
        op_ws_->CreateBlob<float>({num_mean_}, op_name_ + "_mean_value");
    mean_value_->set_data(mean_value.data(), num_mean_);
  }

  void Reshape() override;
  void Forward() override;

 private:
  float scale_;
  int num_mean_;

  BlobF *mean_value_ = nullptr;
};

namespace Vision {

template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, float scale,
                   int num_mean, const T *mean_value, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DATA_OP_HPP
