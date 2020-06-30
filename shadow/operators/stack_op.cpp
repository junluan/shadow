#include "core/operator.hpp"

#include "kernels/stack.hpp"

namespace Shadow {

class StackOp : public Operator {
 public:
  StackOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);

    kernel_ = std::dynamic_pointer_cast<StackKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_GE(inputs.size(), 2);

    const auto& input_0 = inputs[0];
    auto& output = outputs[0];

    int num_axes = input_0->num_axes();
    CHECK(axis_ >= -(num_axes + 1) && axis_ < num_axes + 1)
        << "axis out of bound.";
    if (axis_ < 0) {
      axis_ += num_axes + 1;
    }

    auto out_shape = input_0->shape();
    out_shape.insert(out_shape.begin() + axis_,
                     static_cast<int>(inputs.size()));
    output->reshape(out_shape);

    kernel_->Run(inputs, output, ws_, axis_);
  }

 private:
  int axis_;

  std::shared_ptr<StackKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Stack, StackOp);

}  // namespace Shadow
