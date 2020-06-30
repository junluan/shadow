#include "core/operator.hpp"

namespace Shadow {

class SqueezeOp : public Operator {
 public:
  SqueezeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axes_ = get_repeated_argument<int>("axes");
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int num_axes = input->num_axes();

    VecInt out_shape;
    if (axes_.empty()) {
      for (int n = 0; n < num_axes; ++n) {
        int dim = input->shape(n);
        if (dim > 1) {
          out_shape.push_back(dim);
        } else {
          CHECK_EQ(dim, 1);
        }
      }
    } else {
      for (int n = 0; n < num_axes; ++n) {
        bool need_squeeze = false;
        for (auto axis : axes_) {
          if (n == axis) {
            need_squeeze = true;
            break;
          }
        }
        int dim = input->shape(n);
        if (need_squeeze) {
          CHECK_EQ(dim, 1);
        } else {
          out_shape.push_back(dim);
        }
      }
    }

    output->share_data(input->data<void>(), out_shape);
    CHECK_EQ(output->count(), input->count());
  }

 private:
  VecInt axes_;
};

REGISTER_OPERATOR(Squeeze, SqueezeOp);

}  // namespace Shadow
