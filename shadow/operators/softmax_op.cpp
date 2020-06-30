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

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    axis_ = input->canonical_index(axis_);

    output->reshape(input->shape());

    kernel_->Run(input, output, ws_, axis_);
  }

 private:
  int axis_;

  std::shared_ptr<SoftmaxKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Softmax, SoftmaxOp);

}  // namespace Shadow
