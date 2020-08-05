#include "core/operator.hpp"

#include "kernels/activate.hpp"

namespace Shadow {

class ActivateOp : public Operator {
 public:
  ActivateOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    activate_type_ = get_single_argument<int>("type", 1);
    slope_ = get_single_argument<float>("slope", 0.1);
    CHECK_GE(activate_type_, 0);
    CHECK_LE(activate_type_, 7);

    kernel_ = std::dynamic_pointer_cast<ActivateKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), activate_type_ == kPRelu ? 2 : 1);

    const auto& input = inputs[0];
    auto& output = outputs[0];

    output->reshape(input->shape());

    kernel_->Run(
        input,
        activate_type_ == kPRelu ? inputs[1] : std::shared_ptr<Blob>(nullptr),
        output, ws_, activate_type_, slope_);
  }

 private:
  int activate_type_;
  float slope_;

  std::shared_ptr<ActivateKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Activate, ActivateOp);

}  // namespace Shadow
