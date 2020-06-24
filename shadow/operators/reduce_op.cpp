#include "core/operator.hpp"

#include "kernels/reduce.hpp"

namespace Shadow {

class ReduceOp : public Operator {
 public:
  ReduceOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", 0);
    axes_ = get_repeated_argument<int>("axes");
    keep_dims_ = get_single_argument<bool>("keep_dims", true);

    kernel_ = std::dynamic_pointer_cast<ReduceKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Forward() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    int num_axes = bottom->num_axes();

    auto axes = axes_;
    if (axes.empty()) {
      for (int n = 0; n < num_axes; ++n) {
        axes.push_back(n);
      }
    }

    auto top_shape = bottom->shape();
    for (auto axis : axes) {
      top_shape[bottom->canonical_index(axis)] = 1;
    }
    top->reshape(top_shape);

    kernel_->Run(bottom, top, ws_, operation_, axes);

    if (!keep_dims_) {
      VecInt shape;
      for (int n = 0; n < num_axes; ++n) {
        bool need_squeeze = axes.empty();
        for (auto axis : axes) {
          if (n == axis) {
            need_squeeze = true;
            break;
          }
        }
        int dim = top->shape(n);
        if (need_squeeze) {
          CHECK_EQ(dim, 1);
        } else {
          shape.push_back(dim);
        }
      }
      top->set_shape(shape);
    }
  }

 private:
  int operation_;
  bool keep_dims_;
  VecInt axes_;

  std::shared_ptr<ReduceKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Reduce, ReduceOp);

}  // namespace Shadow
