#include "core/operator.hpp"

#include "kernels/lrn.hpp"

namespace Shadow {

class LRNOp : public Operator {
 public:
  LRNOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    size_ = get_single_argument<int>("local_size", 5);
    CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
    alpha_ = get_single_argument<float>("alpha", 1);
    beta_ = get_single_argument<float>("beta", 0.75);
    norm_region_ = get_single_argument<int>("norm_region", 0);
    CHECK_EQ(norm_region_, 0)
        << "Currently only support norm region method: Across Channels!";
    k_ = get_single_argument<float>("k", 1);

    kernel_ = std::dynamic_pointer_cast<LRNKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    output->reshape(input->shape());

    kernel_->Run(input, output, ws_, size_, alpha_, beta_, k_);
  }

 private:
  int size_, norm_region_;
  float alpha_, beta_, k_;

  std::shared_ptr<LRNKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(LRN, LRNOp);

}  // namespace Shadow
