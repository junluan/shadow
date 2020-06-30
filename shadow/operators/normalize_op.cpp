#include "core/operator.hpp"

#include "kernels/normalize.hpp"

namespace Shadow {

class NormalizeOp : public Operator {
 public:
  NormalizeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    across_spatial_ = get_single_argument<bool>("across_spatial", true);
    channel_shared_ = get_single_argument<bool>("channel_shared", true);
    eps_ = get_single_argument<float>("eps", 1e-5);

    kernel_ = std::dynamic_pointer_cast<NormalizeKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    output->reshape(input->shape());

    if (inputs.size() == 1) {
      kernel_->Run(input, nullptr, output, ws_, across_spatial_,
                   channel_shared_, eps_);
    } else {
      CHECK_EQ(inputs.size(), 2);
      kernel_->Run(input, inputs[1], output, ws_, across_spatial_,
                   channel_shared_, eps_);
    }
  }

 private:
  float eps_;
  bool across_spatial_, channel_shared_;

  std::shared_ptr<NormalizeKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Normalize, NormalizeOp);

}  // namespace Shadow
