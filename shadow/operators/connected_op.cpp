#include "core/operator.hpp"

#include "kernels/connected.hpp"

namespace Shadow {

class ConnectedOp : public Operator {
 public:
  ConnectedOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    CHECK(has_argument("num_output"));
    num_output_ = get_single_argument<int>("num_output", 0);
    bias_term_ = get_single_argument<bool>("bias_term", true);
    transpose_ = get_single_argument<bool>("transpose", true);

    kernel_ = std::dynamic_pointer_cast<ConnectedKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), bias_term_ ? 3 : 2);

    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    output->reshape({input->shape(0), num_output_});

    kernel_->Run(input, weight,
                 bias_term_ ? inputs[2] : std::shared_ptr<Blob>(nullptr),
                 output, ws_, num_output_, bias_term_, transpose_);
  }

 private:
  int num_output_;
  bool bias_term_, transpose_;

  std::shared_ptr<ConnectedKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Connected, ConnectedOp);

}  // namespace Shadow
