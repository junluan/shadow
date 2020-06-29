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

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    int num_axes = bottom->num_axes();
    if (end_axis_ == -1) {
      end_axis_ = num_axes - 1;
    }
    CHECK_LT(end_axis_, num_axes);
    CHECK_LE(axis_, end_axis_);

    VecInt top_shape;
    for (int d = 0; d < axis_; ++d) {
      top_shape.push_back(bottom->shape(d));
    }
    top_shape.push_back(bottom->count(axis_, end_axis_ + 1));
    for (int d = end_axis_ + 1; d < bottom->num_axes(); ++d) {
      top_shape.push_back(bottom->shape(d));
    }

    top->share_data(bottom->data<float>(), top_shape);
    CHECK_EQ(top->count(), bottom->count());
  }

 private:
  int axis_, end_axis_;
};

REGISTER_OPERATOR(Flatten, FlattenOp);

}  // namespace Shadow
