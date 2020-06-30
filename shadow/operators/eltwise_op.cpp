#include "core/operator.hpp"

#include "kernels/eltwise.hpp"

namespace Shadow {

class EltwiseOp : public Operator {
 public:
  EltwiseOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    // Prod: 0, Sum: 1, Max: 2, Min: 3
    operation_ = get_single_argument<int>("operation", -1);
    CHECK_GE(operation_, 0);
    CHECK_LE(operation_, 3);
    coeff_ = get_repeated_argument<float>("coeff");

    kernel_ = std::dynamic_pointer_cast<EltwiseKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_GE(inputs.size(), 2);

    const auto& input_0 = inputs[0];
    auto& output = outputs[0];

    for (int n = 1; n < inputs.size(); ++n) {
      CHECK(inputs[n]->shape() == input_0->shape());
    }
    output->reshape(input_0->shape());

    int coeff_size = static_cast<int>(coeff_.size());

    CHECK(coeff_size == 0 || coeff_size == inputs.size())
        << "Eltwise op takes one coefficient per input blob.";
    CHECK(!(operation_ != 1 && coeff_size))
        << "Eltwise op only takes coefficients for summation.";

    VecFloat coeff(inputs.size(), 1);
    for (int n = 0; n < coeff_size; ++n) {
      coeff[n] = coeff_[n];
    }

    kernel_->Run(inputs, output, ws_, operation_, coeff);
  }

 private:
  int operation_;
  VecFloat coeff_;

  std::shared_ptr<EltwiseKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Eltwise, EltwiseOp);

}  // namespace Shadow
