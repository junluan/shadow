#include "core/operator.hpp"

namespace Shadow {

class ReshapeOp : public Operator {
 public:
  ReshapeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);
    num_axes_ = get_single_argument<int>("num_axes", -1);
    CHECK_GE(num_axes_, -1);
    shape_ = get_repeated_argument<int>("shape");
  }

  void Run() override {
    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    int inferred_axis = -1, constant_count = 1;
    VecInt copy_axes;
    for (int d = 0; d < shape_.size(); ++d) {
      int top_dim = shape_[d];
      if (top_dim == 0) {
        copy_axes.push_back(d);
      } else if (top_dim == -1) {
        CHECK_EQ(inferred_axis, -1);
        inferred_axis = d;
      } else {
        constant_count *= top_dim;
      }
    }

    int start_axis = bottom->canonical_index(axis_);
    CHECK_GE(start_axis, 0);
    CHECK_LT(start_axis, bottom->num_axes());
    int end_axis =
        (num_axes_ == -1) ? bottom->num_axes() : (start_axis + num_axes_);
    CHECK_LE(end_axis, bottom->num_axes());
    int num_axes_replaced = end_axis - start_axis;
    int num_axes_retained = bottom->num_axes() - num_axes_replaced;

    VecInt top_shape(num_axes_retained + shape_.size());
    int top_shape_index = 0;
    for (int d = 0; d < start_axis; ++d) {
      top_shape[top_shape_index++] = bottom->shape(d);
    }
    for (auto dim : shape_) {
      top_shape[top_shape_index++] = dim;
    }
    for (int d = end_axis; d < bottom->num_axes(); ++d) {
      top_shape[top_shape_index++] = bottom->shape(d);
    }
    CHECK_EQ(top_shape_index, top_shape.size());

    for (auto d : copy_axes) {
      CHECK_GT(bottom->num_axes(), start_axis + d);
      top_shape[start_axis + d] = bottom->shape(start_axis + d);
    }

    if (inferred_axis >= 0) {
      int explicit_count = constant_count;
      explicit_count *= bottom->count(0, start_axis);
      explicit_count *= bottom->count(end_axis);
      for (auto d : copy_axes) {
        explicit_count *= top_shape[start_axis + d];
      }
      CHECK_EQ(0, bottom->count() % explicit_count);
      top_shape[start_axis + inferred_axis] = bottom->count() / explicit_count;
    }

    top->share_data(bottom->data<float>(), top_shape);
    CHECK_EQ(top->count(), bottom->count());
  }

 private:
  int axis_, num_axes_;
  VecInt shape_;
};

REGISTER_OPERATOR(Reshape, ReshapeOp);

}  // namespace Shadow
