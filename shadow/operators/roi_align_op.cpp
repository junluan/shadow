#include "core/operator.hpp"

#include "kernels/roi_align.hpp"

namespace Shadow {

class ROIAlignOp : public Operator {
 public:
  ROIAlignOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    pooled_h_ = get_single_argument<int>("pooled_h", 0);
    pooled_w_ = get_single_argument<int>("pooled_w", 0);
    CHECK_GT(pooled_h_, 1) << "pooled_h must be > 1";
    CHECK_GT(pooled_w_, 1) << "pooled_w must be > 1";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);

    kernel_ = std::dynamic_pointer_cast<ROIAlignKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Forward() override {
    CHECK_EQ(bottoms_size(), 2);

    const auto bottom = bottoms(0);
    const auto roi = bottoms(1);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    top->reshape({roi->shape(0), bottom->shape(1), pooled_h_, pooled_w_});

    kernel_->Run(bottom, roi, top, ws_, pooled_h_, pooled_w_, spatial_scale_);
  }

 private:
  int pooled_h_, pooled_w_;
  float spatial_scale_;

  std::shared_ptr<ROIAlignKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(ROIAlign, ROIAlignOp);

}  // namespace Shadow
