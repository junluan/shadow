#include "core/operator.hpp"

#include "kernels/axpy.hpp"

namespace Shadow {

/**
 * @param Formulation:
 *            F = a * X + Y
 *        Shape info:
 *            a:  N x C          --> inputs[0]
 *            X:  N x C x H x W  --> inputs[1]
 *            Y:  N x C x H x W  --> inputs[2]
 *            F:  N x C x H x W  --> outputs[0]
 */
class AxpyOp : public Operator {
 public:
  AxpyOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    kernel_ = std::dynamic_pointer_cast<AxpyKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), 3)
        << "This op must have three inputs: alpha, x and y";

    const auto& alpha = inputs[0];
    const auto& x = inputs[1];
    const auto& y = inputs[2];
    auto& output = outputs[0];

    CHECK_EQ(alpha->shape(0), x->shape(0));
    CHECK_EQ(alpha->shape(1), x->shape(1));
    if (alpha->num_axes() == 4) {
      CHECK_EQ(alpha->shape(2), 1);
      CHECK_EQ(alpha->shape(3), 1);
    }
    CHECK(x->shape() == y->shape());

    output->reshape(x->shape());

    kernel_->Run(alpha, x, y, output, ws_);
  }

 private:
  std::shared_ptr<AxpyKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Axpy, AxpyOp);

}  // namespace Shadow
