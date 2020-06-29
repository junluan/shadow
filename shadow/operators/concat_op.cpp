#include "core/operator.hpp"

#include "kernels/concat.hpp"

namespace Shadow {

class ConcatOp : public Operator {
 public:
  ConcatOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);

    kernel_ = std::dynamic_pointer_cast<ConcatKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run() override {
    CHECK_GE(bottoms_size(), 2);

    const auto bottom_0 = bottoms(0);
    auto top = tops(0);

    axis_ = bottom_0->canonical_index(axis_);

    std::vector<std::shared_ptr<Blob>> bottom_blobs(1, bottom_0);

    auto top_shape = bottom_0->shape();
    int num_axes = bottom_0->num_axes();
    for (int n = 1; n < bottoms_size(); ++n) {
      const auto bottom = bottoms(n);
      CHECK_EQ(num_axes, bottom->num_axes())
          << "Bottoms must have the same axes!";
      for (int d = 0; d < num_axes; ++d) {
        if (d != axis_) {
          CHECK_EQ(top_shape[d], bottom->shape(d))
              << "Bottoms must have the same shape, except at concat_axis!";
        }
      }
      top_shape[axis_] += bottom->shape(axis_);
      bottom_blobs.push_back(bottom);
    }

    top->reshape(top_shape);

    kernel_->Run(bottom_blobs, top, ws_, axis_);
  }

 private:
  int axis_;

  std::shared_ptr<ConcatKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Concat, ConcatOp);

}  // namespace Shadow
