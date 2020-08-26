#include "core/operator.hpp"

#include "kernels/reduce.hpp"

namespace Shadow {

class ReduceOp : public Operator {
 public:
  ReduceOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", 0);
    CHECK_GE(operation_, 0);
    CHECK_LE(operation_, 6);
    axes_ = get_repeated_argument<int>("axes");
    keep_dims_ = get_single_argument<bool>("keep_dims", true);

    kernel_ = std::dynamic_pointer_cast<ReduceKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int num_axes = input->num_axes();

    auto axes = axes_;
    if (axes.empty()) {
      for (int n = 0; n < num_axes; ++n) {
        axes.push_back(n);
      }
    }

    auto out_shape = input->shape();
    for (auto axis : axes) {
      out_shape[input->canonical_index(axis)] = 1;
    }
    output->reshape(out_shape);

    kernel_->Run(input, output, ws_, operation_, axes);

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
        int dim = output->shape(n);
        if (need_squeeze) {
          CHECK_EQ(dim, 1);
        } else {
          shape.push_back(dim);
        }
      }
      output->set_shape(shape);
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
