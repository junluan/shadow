#include "core/operator.hpp"

#include "kernels/pad.hpp"

namespace Shadow {

class PadOp : public Operator {
 public:
  PadOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    // [top, bottom, left, right]
    paddings_ = get_repeated_argument<int>("paddings");
    CHECK_EQ(paddings_.size(), 4);
    value_ = get_single_argument<float>("value", 0);

    kernel_ = std::dynamic_pointer_cast<PadKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    auto out_shape = input->shape();
    out_shape[2] = input->shape(2) + paddings_[0] + paddings_[1];
    out_shape[3] = input->shape(3) + paddings_[2] + paddings_[3];
    output->reshape(out_shape);

    kernel_->Run(input, output, ws_, paddings_, value_);
  }

 private:
  float value_;
  VecInt paddings_;

  std::shared_ptr<PadKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Pad, PadOp);

}  // namespace Shadow
