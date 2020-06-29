#include "core/operator.hpp"

#include "kernels/roi_pooling.hpp"

namespace Shadow {

class ROIPoolingOp : public Operator {
 public:
  ROIPoolingOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    pooled_h_ = get_single_argument<int>("pooled_h", 0);
    pooled_w_ = get_single_argument<int>("pooled_w", 0);
    CHECK_GT(pooled_h_, 0) << "pooled_h must be > 0";
    CHECK_GT(pooled_w_, 0) << "pooled_w must be > 0";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);

    kernel_ = std::dynamic_pointer_cast<ROIPoolingKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run() override {
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

  std::shared_ptr<ROIPoolingKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(ROIPooling, ROIPoolingOp);

}  // namespace Shadow
