#include "core/operator.hpp"

#include "kernels/axpy.hpp"

namespace Shadow {

/**
 * @param Formulation:
 *            F = a * X + Y
 *        Shape info:
 *            a:  N x C          --> bottoms[0]
 *            X:  N x C x H x W  --> bottoms[1]
 *            Y:  N x C x H x W  --> bottoms[2]
 *            F:  N x C x H x W  --> tops[0]
 */
class AxpyOp : public Operator {
 public:
  AxpyOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    kernel_ = std::dynamic_pointer_cast<AxpyKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run() override {
    CHECK_EQ(bottoms_size(), 3)
        << "This op must have three bottoms: alpha, x and y";

    const auto alpha = bottoms(0);
    const auto x = bottoms(1);
    const auto y = bottoms(2);
    auto top = tops(0);

    CHECK_EQ(alpha->shape(0), x->shape(0));
    CHECK_EQ(alpha->shape(1), x->shape(1));
    if (alpha->num_axes() == 4) {
      CHECK_EQ(alpha->shape(2), 1);
      CHECK_EQ(alpha->shape(3), 1);
    }
    CHECK(x->shape() == y->shape());

    top->reshape(x->shape());

    kernel_->Run(alpha, x, y, top, ws_);
  }

 private:
  std::shared_ptr<AxpyKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Axpy, AxpyOp);

}  // namespace Shadow
