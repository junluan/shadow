#include "core/operator.hpp"

#include "kernels/deform_psroi_pooling.hpp"

namespace Shadow {

class DeformPSROIPoolingOp : public Operator {
 public:
  DeformPSROIPoolingOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    output_dim_ = get_single_argument<int>("output_dim", 0);
    group_size_ = get_single_argument<int>("group_size", 0);
    pooled_size_ = get_single_argument<int>("pooled_size", 0);
    part_size_ = get_single_argument<int>("part_size", 0);
    sample_per_part_ = get_single_argument<int>("sample_per_part", 1);
    CHECK_GT(output_dim_, 0) << "output_dim must be > 0";
    CHECK_GT(group_size_, 0) << "group_size must be > 0";
    CHECK_GT(pooled_size_, 0) << "pooled_size must be > 0";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);
    trans_std_ = get_single_argument<float>("trans_std", 0);
    no_trans_ = get_single_argument<bool>("no_trans", false);
    if (part_size_ == 0) {
      part_size_ = pooled_size_;
    }

    kernel_ = std::dynamic_pointer_cast<DeformPSROIPoolingKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), no_trans_ ? 2 : 3);

    const auto& input = inputs[0];
    const auto& roi = inputs[1];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    output->reshape({roi->shape(0), output_dim_, pooled_size_, pooled_size_});

    kernel_->Run(input, roi,
                 no_trans_ ? std::shared_ptr<Blob>(nullptr) : inputs[2], output,
                 ws_, output_dim_, group_size_, pooled_size_, part_size_,
                 sample_per_part_, spatial_scale_, trans_std_, no_trans_);
  }

 private:
  int output_dim_, group_size_, pooled_size_, part_size_, sample_per_part_;
  float spatial_scale_, trans_std_;
  bool no_trans_;

  std::shared_ptr<DeformPSROIPoolingKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(DeformPSROIPooling, DeformPSROIPoolingOp);

}  // namespace Shadow
