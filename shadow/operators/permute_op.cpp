#include "core/operator.hpp"

#include "kernels/permute.hpp"

namespace Shadow {

class PermuteOp : public Operator {
 public:
  PermuteOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    order_value_ = get_repeated_argument<int>("order");

    kernel_ = std::dynamic_pointer_cast<PermuteKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    CHECK_EQ(bottom->num_axes(), order_value_.size());

    VecInt top_shape;
    for (const auto& axis : order_value_) {
      top_shape.push_back(bottom->shape(axis));
    }
    top->reshape(top_shape);

    kernel_->Run(bottom, top, ws_, order_value_);
  }

 private:
  VecInt order_value_;

  std::shared_ptr<PermuteKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Permute, PermuteOp);

}  // namespace Shadow
