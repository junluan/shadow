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

  void Forward() override {
    CHECK_EQ(bottoms_size(), bias_term_ ? 3 : 2);

    const auto bottom = bottoms(0);
    const auto weight = bottoms(1);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    top->reshape({bottom->shape(0), num_output_});

    kernel_->Run(bottom, weight, bias_term_ ? bottoms(2) : nullptr, top, ws_,
                 num_output_, bias_term_, transpose_);
  }

 private:
  int num_output_;
  bool bias_term_, transpose_;

  std::shared_ptr<ConnectedKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Connected, ConnectedOp);

}  // namespace Shadow
