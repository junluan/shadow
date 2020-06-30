#include "core/operator.hpp"

#include "kernels/group_norm.hpp"

namespace Shadow {

class GroupNormOp : public Operator {
 public:
  GroupNormOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    group_ = get_single_argument<int>("group", 1);
    eps_ = get_single_argument<float>("eps", 1e-5);

    kernel_ = std::dynamic_pointer_cast<GroupNormKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    output->reshape(input->shape());

    CHECK_EQ(input->shape(1) % group_, 0);

    if (inputs.size() == 1) {
      kernel_->Run(input, nullptr, nullptr, output, ws_, group_, eps_);
    } else {
      CHECK_EQ(inputs.size(), 3);
      kernel_->Run(input, inputs[1], inputs[2], output, ws_, group_, eps_);
    }
  }

 private:
  int group_;
  float eps_;

  std::shared_ptr<GroupNormKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(GroupNorm, GroupNormOp);

}  // namespace Shadow
