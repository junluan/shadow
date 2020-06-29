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

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    top->reshape(bottom->shape());

    int channel = bottom->shape(1);

    if (bottoms_size() == 1) {
      kernel_->Run(bottom, nullptr, nullptr, top, ws_, channel, eps_);
    } else {
      CHECK_EQ(bottoms_size(), 3);
      kernel_->Run(bottom, bottoms(1), bottoms(2), top, ws_, channel, eps_);
    }
  }

 private:
  float eps_;

  std::shared_ptr<GroupNormKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(InstanceNorm, InstanceNormOp);

}  // namespace Shadow
