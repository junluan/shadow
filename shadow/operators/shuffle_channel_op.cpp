#include "core/operator.hpp"

#include "kernels/shuffle_channel.hpp"

namespace Shadow {

class ShuffleChannelOp : public Operator {
 public:
  ShuffleChannelOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    group_ = get_single_argument<int>("group", 0);
    CHECK_GT(group_, 0) << "group must be larger than 0";

    kernel_ = std::dynamic_pointer_cast<ShuffleChannelKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    CHECK_EQ(input->shape(1) % group_, 0);

    output->reshape(input->shape());

    kernel_->Run(input, output, ws_, group_);
  }

 private:
  int group_;

  std::shared_ptr<ShuffleChannelKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(ShuffleChannel, ShuffleChannelOp);

}  // namespace Shadow
