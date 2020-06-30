#include "core/operator.hpp"

#include "kernels/scale.hpp"

namespace Shadow {

class ScaleOp : public Operator {
 public:
  ScaleOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);
    has_scale_ = get_single_argument<bool>("has_scale", false);
    has_bias_ = get_single_argument<bool>("has_bias", false);
    scale_value_ = get_repeated_argument<float>("scale_value");
    bias_value_ = get_repeated_argument<float>("bias_value");

    kernel_ = std::dynamic_pointer_cast<ScaleKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    output->reshape(input->shape());

    if (has_scale_ && has_bias_) {
      CHECK_EQ(inputs.size(), 3);
      kernel_->Run(input, inputs[1], inputs[2], output, ws_, axis_);
    } else if (has_scale_) {
      CHECK_EQ(inputs.size(), 2);
      kernel_->Run(input, inputs[1], nullptr, output, ws_, axis_);
    } else if (has_bias_) {
      CHECK_EQ(inputs.size(), 2);
      kernel_->Run(input, nullptr, inputs[1], output, ws_, axis_);
    } else {
      int dim = input->shape(axis_);
      if (scale_value_.size() > 1) {
        CHECK_EQ(scale_value_.size(), dim);
      } else if (scale_value_.size() == 1) {
        scale_value_ = VecFloat(dim, scale_value_[0]);
      }
      if (bias_value_.size() > 1) {
        CHECK_EQ(bias_value_.size(), dim);
      } else if (bias_value_.size() == 1) {
        bias_value_ = VecFloat(dim, bias_value_[0]);
      }
      kernel_->Run(input, output, ws_, axis_, scale_value_, bias_value_);
    }
  }

 private:
  int axis_;
  bool has_scale_, has_bias_;
  VecFloat scale_value_, bias_value_;

  std::shared_ptr<ScaleKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Scale, ScaleOp);

}  // namespace Shadow
