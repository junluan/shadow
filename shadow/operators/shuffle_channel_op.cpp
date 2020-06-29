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

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    CHECK_EQ(bottom->shape(1) % group_, 0);

    top->reshape(bottom->shape());

    kernel_->Run(bottom, top, ws_, group_);
  }

 private:
  int group_;

  std::shared_ptr<ShuffleChannelKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(ShuffleChannel, ShuffleChannelOp);

}  // namespace Shadow
