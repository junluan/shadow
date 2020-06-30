#include "core/operator.hpp"

namespace Shadow {

class FlattenOp : public Operator {
 public:
  FlattenOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);
    CHECK_GE(axis_, 0);
    end_axis_ = get_single_argument<int>("end_axis", -1);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int num_axes = input->num_axes();
    if (end_axis_ == -1) {
      end_axis_ = num_axes - 1;
    }
    CHECK_LT(end_axis_, num_axes);
    CHECK_LE(axis_, end_axis_);

    VecInt out_shape;
    for (int d = 0; d < axis_; ++d) {
      out_shape.push_back(input->shape(d));
    }
    out_shape.push_back(input->count(axis_, end_axis_ + 1));
    for (int d = end_axis_ + 1; d < input->num_axes(); ++d) {
      out_shape.push_back(input->shape(d));
    }

    output->share_data(input->data<void>(), out_shape);
    CHECK_EQ(output->count(), input->count());
  }

 private:
  int axis_, end_axis_;
};

REGISTER_OPERATOR(Flatten, FlattenOp);

}  // namespace Shadow
