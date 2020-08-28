#include "core/operator.hpp"

#include "kernels/normalize.hpp"

namespace Shadow {

class NormalizeOp : public Operator {
 public:
  NormalizeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    p_ = get_single_argument<float>("p", 2);
    axis_ = get_single_argument<int>("axis", 1);
    eps_ = get_single_argument<float>("eps", 1e-12);

    kernel_ = std::dynamic_pointer_cast<NormalizeKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    output->reshape(input->shape());

    kernel_->Run(input, output, ws_, p_, input->canonical_index(axis_), eps_);
  }

 private:
  float p_, eps_;
  int axis_;

  std::shared_ptr<NormalizeKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Normalize, NormalizeOp);

}  // namespace Shadow
