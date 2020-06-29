#include "core/operator.hpp"

#include "kernels/stack.hpp"

namespace Shadow {

class StackOp : public Operator {
 public:
  StackOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);

    kernel_ = std::dynamic_pointer_cast<StackKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run() override {
    CHECK_GE(bottoms_size(), 2);

    const auto bottom_0 = bottoms(0);
    auto top = tops(0);

    int num_axes = bottom_0->num_axes();
    CHECK(axis_ >= -(num_axes + 1) && axis_ < num_axes + 1)
        << "axis out of bound.";
    if (axis_ < 0) {
      axis_ += num_axes + 1;
    }

    auto top_shape = bottom_0->shape();
    top_shape.insert(top_shape.begin() + axis_, bottoms_size());
    top->reshape(top_shape);

    std::vector<std::shared_ptr<Blob>> bottom_blobs;
    for (int n = 0; n < bottoms_size(); ++n) {
      bottom_blobs.push_back(bottoms(n));
    }

    kernel_->Run(bottom_blobs, top, ws_, axis_);
  }

 private:
  int axis_;

  std::shared_ptr<StackKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Stack, StackOp);

}  // namespace Shadow
