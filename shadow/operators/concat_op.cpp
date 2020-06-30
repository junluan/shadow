#include "core/operator.hpp"

#include "kernels/concat.hpp"

namespace Shadow {

class ConcatOp : public Operator {
 public:
  ConcatOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);

    kernel_ = std::dynamic_pointer_cast<ConcatKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_GE(inputs.size(), 2);

    const auto& input_0 = inputs[0];
    auto& output = outputs[0];

    axis_ = input_0->canonical_index(axis_);

    auto in_shape = input_0->shape();
    int num_axes = input_0->num_axes();
    for (int n = 1; n < inputs.size(); ++n) {
      const auto& input = inputs[n];
      CHECK_EQ(num_axes, input->num_axes())
          << "Inputs must have the same axes!";
      for (int d = 0; d < num_axes; ++d) {
        if (d != axis_) {
          CHECK_EQ(in_shape[d], input->shape(d))
              << "Inputs must have the same shape, except at concat_axis!";
        }
      }
      in_shape[axis_] += input->shape(axis_);
    }

    output->reshape(in_shape);

    kernel_->Run(inputs, output, ws_, axis_);
  }

 private:
  int axis_;

  std::shared_ptr<ConcatKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Concat, ConcatOp);

}  // namespace Shadow
