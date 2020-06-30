#include "core/operator.hpp"

#include "kernels/reorg.hpp"

namespace Shadow {

class ReorgOp : public Operator {
 public:
  ReorgOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    stride_ = get_single_argument<int>("stride", 2);

    kernel_ = std::dynamic_pointer_cast<ReorgKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int in_c = input->shape(1), in_h = input->shape(2), in_w = input->shape(3);

    CHECK_EQ(in_h % stride_, 0);
    CHECK_EQ(in_w % stride_, 0);

    auto out_shape = input->shape();
    out_shape[1] = in_c * stride_ * stride_;
    out_shape[2] = in_h / stride_;
    out_shape[3] = in_w / stride_;
    output->reshape(out_shape);

    kernel_->Run(input, output, ws_, stride_);
  }

 private:
  int stride_;

  std::shared_ptr<ReorgKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Reorg, ReorgOp);

}  // namespace Shadow
