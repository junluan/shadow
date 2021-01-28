#include "core/operator.hpp"

#include "kernels/psroi_pooling.hpp"

namespace Shadow {

class PSROIPoolingOp : public Operator {
 public:
  PSROIPoolingOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    pooled_h_ = get_single_argument<int>("pooled_h", 0);
    pooled_w_ = get_single_argument<int>("pooled_w", 0);
    CHECK_GT(pooled_h_, 0) << "pooled_h must be > 0";
    CHECK_GT(pooled_w_, 0) << "pooled_w must be > 0";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);

    kernel_ = std::dynamic_pointer_cast<PSROIPoolingKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), 2);

    const auto& input = inputs[0];
    const auto& roi = inputs[1];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int in_c = input->shape(1), out_spatial_dim = pooled_h_ * pooled_w_;

    CHECK_EQ(in_c % out_spatial_dim, 0);

    output->reshape(
        {roi->shape(0), in_c / out_spatial_dim, pooled_h_, pooled_w_});

    kernel_->Run(input, roi, output, ws_, spatial_scale_);
  }

 private:
  int pooled_h_, pooled_w_;
  float spatial_scale_;

  std::shared_ptr<PSROIPoolingKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(PSROIPooling, PSROIPoolingOp);

}  // namespace Shadow
