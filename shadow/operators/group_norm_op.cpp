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

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    top->reshape(bottom->shape());

    CHECK_EQ(bottom->shape(1) % group_, 0);

    if (bottoms_size() == 1) {
      kernel_->Run(bottom, nullptr, nullptr, top, ws_, group_, eps_);
    } else {
      CHECK_EQ(bottoms_size(), 3);
      kernel_->Run(bottom, bottoms(1), bottoms(2), top, ws_, group_, eps_);
    }
  }

 private:
  int group_;
  float eps_;

  std::shared_ptr<GroupNormKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(GroupNorm, GroupNormOp);

}  // namespace Shadow
