#include "core/operator.hpp"

namespace Shadow {

class UnsqueezeOp : public Operator {
 public:
  UnsqueezeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axes_ = get_repeated_argument<int>("axes");
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int num_axes = input->num_axes();

    VecInt axes;
    for (auto axis : axes_) {
      if (axis < 0) {
        axis += num_axes + 1;
      }
      axes.push_back(axis);
    }

    VecInt out_shape(num_axes + axes.size(), 0);
    for (auto axis : axes) {
      out_shape[axis] = 1;
    }
    int d = 0;
    for (auto& dim : out_shape) {
      if (dim == 0) {
        dim = input->shape(d++);
      }
    }
    CHECK_EQ(d, num_axes);

    output->share_data(input->data<void>(), out_shape);
    CHECK_EQ(output->count(), input->count());
  }

 private:
  VecInt axes_;
};

REGISTER_OPERATOR(Unsqueeze, UnsqueezeOp);

}  // namespace Shadow
