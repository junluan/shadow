#include "core/operator.hpp"

#include "kernels/pad.hpp"

namespace Shadow {

class PadOp : public Operator {
 public:
  PadOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    // [top, bottom, left, right]
    paddings_ = get_repeated_argument<int>("paddings");
    CHECK_EQ(paddings_.size(), 4);
    value_ = get_single_argument<float>("value", 0);

    kernel_ = std::dynamic_pointer_cast<PadKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    auto top_shape = bottom->shape();
    top_shape[2] = bottom->shape(2) + paddings_[0] + paddings_[1];
    top_shape[3] = bottom->shape(3) + paddings_[2] + paddings_[3];
    top->reshape(top_shape);

    kernel_->Run(bottom, top, ws_, paddings_, value_);
  }

 private:
  float value_;
  VecInt paddings_;

  std::shared_ptr<PadKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Pad, PadOp);

}  // namespace Shadow
