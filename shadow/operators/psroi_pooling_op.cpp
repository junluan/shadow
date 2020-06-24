#include "core/operator.hpp"

#include "kernels/psroi_pooling.hpp"

namespace Shadow {

class PSROIPoolingOp : public Operator {
 public:
  PSROIPoolingOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    output_dim_ = get_single_argument<int>("output_dim", 0);
    group_size_ = get_single_argument<int>("group_size", 0);
    CHECK_GT(output_dim_, 0) << "output_dim must be > 0";
    CHECK_GT(group_size_, 0) << "group_size must be > 0";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);
    pooled_h_ = group_size_, pooled_w_ = group_size_;

    kernel_ = std::dynamic_pointer_cast<PSROIPoolingKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Forward() override {
    CHECK_EQ(bottoms_size(), 2);

    const auto bottom = bottoms(0);
    const auto roi = bottoms(1);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    top->reshape({bottom->shape(0), output_dim_, pooled_h_, pooled_w_});

    kernel_->Run(bottom, roi, top, ws_, output_dim_, group_size_, pooled_h_,
                 pooled_w_, spatial_scale_);
  }

 private:
  int output_dim_, group_size_, pooled_h_, pooled_w_;
  float spatial_scale_;

  std::shared_ptr<PSROIPoolingKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(PSROIPooling, PSROIPoolingOp);

}  // namespace Shadow
