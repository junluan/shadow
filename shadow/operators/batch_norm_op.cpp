#include "core/operator.hpp"

#include "kernels/batch_norm.hpp"

namespace Shadow {

class BatchNormOp : public Operator {
 public:
  BatchNormOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    CHECK(get_single_argument<bool>("use_global_stats", true));
    eps_ = get_single_argument<float>("eps", 1e-5);

    kernel_ = std::dynamic_pointer_cast<BatchNormKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_GE(inputs.size(), 3);

    const auto& input = inputs[0];
    const auto& mean = inputs[1];
    const auto& variance = inputs[2];
    auto& output = outputs[0];

    output->reshape(input->shape());

    float scale_factor = 1;
    if (inputs.size() == 4) {
      float scale = 1;
      CHECK_EQ(inputs[3]->count(), 1);
      inputs[3]->get_data<float>(&scale, 1);
      scale_factor = scale == 0 ? 0 : 1 / scale;
    }

    kernel_->Run(input, mean, variance, output, ws_, scale_factor, eps_);
  }

 private:
  float eps_;

  std::shared_ptr<BatchNormKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(BatchNorm, BatchNormOp);

}  // namespace Shadow
