#include "core/operator.hpp"

#include "kernels/reorg.hpp"

namespace Shadow {

class ReorgOp : public Operator {
 public:
  ReorgOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    stride_ = get_single_argument<int>("stride", 2);

    kernel_ = std::dynamic_pointer_cast<ReorgKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Forward() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    int in_c = bottom->shape(1), in_h = bottom->shape(2),
        in_w = bottom->shape(3);

    CHECK_EQ(in_h % stride_, 0);
    CHECK_EQ(in_w % stride_, 0);

    auto top_shape = bottom->shape();
    top_shape[1] = in_c * stride_ * stride_;
    top_shape[2] = in_h / stride_;
    top_shape[3] = in_w / stride_;
    top->reshape(top_shape);

    kernel_->Run(bottom, top, ws_, stride_);
  }

 private:
  int stride_;

  std::shared_ptr<ReorgKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Reorg, ReorgOp);

}  // namespace Shadow
