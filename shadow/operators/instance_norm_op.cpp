#include "core/operator.hpp"

#include "kernels/group_norm.hpp"

namespace Shadow {

class InstanceNormOp : public Operator {
 public:
  InstanceNormOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    eps_ = get_single_argument<float>("eps", 1e-5);

    kernel_ = std::dynamic_pointer_cast<GroupNormKernel>(
        CreateKernel("GroupNorm", ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    output->reshape(input->shape());

    int channel = input->shape(1);

    if (inputs.size() == 1) {
      kernel_->Run(input, nullptr, nullptr, output, ws_, channel, eps_);
    } else {
      CHECK_EQ(inputs.size(), 3);
      kernel_->Run(input, inputs[1], inputs[2], output, ws_, channel, eps_);
    }
  }

 private:
  float eps_;

  std::shared_ptr<GroupNormKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(InstanceNorm, InstanceNormOp);

}  // namespace Shadow
