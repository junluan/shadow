#include "core/operator.hpp"

#include "kernels/softmax.hpp"

namespace Shadow {

class SoftmaxOp : public Operator {
 public:
  SoftmaxOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);

    kernel_ = std::dynamic_pointer_cast<SoftmaxKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    axis_ = bottom->canonical_index(axis_);

    top->reshape(bottom->shape());

    kernel_->Run(bottom, top, ws_, axis_);
  }

 private:
  int axis_;

  std::shared_ptr<SoftmaxKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Softmax, SoftmaxOp);

}  // namespace Shadow
