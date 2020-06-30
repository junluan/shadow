#include "core/operator.hpp"

#include "kernels/permute.hpp"

namespace Shadow {

class PermuteOp : public Operator {
 public:
  PermuteOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    order_value_ = get_repeated_argument<int>("order");

    kernel_ = std::dynamic_pointer_cast<PermuteKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    CHECK_EQ(input->num_axes(), order_value_.size());

    VecInt out_shape;
    for (const auto& axis : order_value_) {
      out_shape.push_back(input->shape(axis));
    }
    output->reshape(out_shape);

    kernel_->Run(input, output, ws_, order_value_);
  }

 private:
  VecInt order_value_;

  std::shared_ptr<PermuteKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Permute, PermuteOp);

}  // namespace Shadow
