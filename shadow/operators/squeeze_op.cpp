#include "core/operator.hpp"

namespace Shadow {

class SqueezeOp : public Operator {
 public:
  SqueezeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axes_ = get_repeated_argument<int>("axes");
  }

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    int num_axes = bottom->num_axes();

    VecInt top_shape;
    if (axes_.empty()) {
      for (int n = 0; n < num_axes; ++n) {
        int dim = bottom->shape(n);
        if (dim > 1) {
          top_shape.push_back(dim);
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
        int dim = bottom->shape(n);
        if (need_squeeze) {
          CHECK_EQ(dim, 1);
        } else {
          top_shape.push_back(dim);
        }
      }
    }

    top->share_data(bottom->data<float>(), top_shape);
    CHECK_EQ(top->count(), bottom->count());
  }

 private:
  VecInt axes_;
};

REGISTER_OPERATOR(Squeeze, SqueezeOp);

}  // namespace Shadow
