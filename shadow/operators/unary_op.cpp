#include "core/operator.hpp"

#include "kernels/unary.hpp"

namespace Shadow {

class UnaryOp : public Operator {
 public:
  UnaryOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", -1);
    CHECK_GE(operation_, 0);
    CHECK_LE(operation_, 14);

    kernel_ = std::dynamic_pointer_cast<UnaryKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    top->reshape(bottom->shape());

    kernel_->Run(bottom, top, ws_, operation_);
  }

 private:
  int operation_;

  std::shared_ptr<UnaryKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Unary, UnaryOp);

}  // namespace Shadow
