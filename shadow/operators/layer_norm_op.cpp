#include "core/operator.hpp"

#include "kernels/layer_norm.hpp"

namespace Shadow {

class LayerNormOp : public Operator {
 public:
  LayerNormOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    normalized_shape_ = get_repeated_argument<int>("normalized_shape");
    CHECK(!normalized_shape_.empty());
    eps_ = get_single_argument<float>("eps", 1e-5);

    kernel_ = std::dynamic_pointer_cast<LayerNormKernel>(
        CreateKernel("LayerNorm", ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    output->reshape(input->shape());

    for (int n = 0; n < normalized_shape_.size(); ++n) {
      CHECK_EQ(input->shape(n - normalized_shape_.size()),
               normalized_shape_[n]);
    }

    if (inputs.size() == 1) {
      kernel_->Run(input, nullptr, nullptr, output, ws_, normalized_shape_,
                   eps_);
    } else {
      CHECK_EQ(inputs.size(), 3);
      kernel_->Run(input, inputs[1], inputs[2], output, ws_, normalized_shape_,
                   eps_);
    }
  }

 private:
  float eps_;
  VecInt normalized_shape_;

  std::shared_ptr<LayerNormKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(LayerNorm, LayerNormOp);

}  // namespace Shadow
