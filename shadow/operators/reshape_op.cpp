#include "reshape_op.hpp"

namespace Shadow {

void ReshapeOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int start_axis = (axis_ >= 0) ? axis_ : (bottom->num_axes() + axis_ + 1);
  CHECK_GE(start_axis, 0);
  CHECK_LE(start_axis, bottom->num_axes());
  int end_axis =
      (num_axes_ == -1) ? bottom->num_axes() : (start_axis + num_axes_);
  CHECK_LE(end_axis, bottom->num_axes());
  int num_axes_replaced = end_axis - start_axis;
  int num_axes_retained = bottom->num_axes() - num_axes_replaced;
  VecInt top_shape(num_axes_retained + shape_.size());
  int top_shape_index = 0;
  for (int i = 0; i < start_axis; ++i) {
    top_shape[top_shape_index++] = bottom->shape(i);
  }
  for (const auto shape_dim : shape_) {
    top_shape[top_shape_index++] = shape_dim;
  }
  for (int i = end_axis; i < bottom->num_axes(); ++i) {
    top_shape[top_shape_index++] = bottom->shape(i);
  }
  CHECK_EQ(top_shape_index, top_shape.size());
  for (const auto copy_axis : copy_axes_) {
    CHECK_GT(bottom->num_axes(), start_axis + copy_axis);
    top_shape[start_axis + copy_axis] = bottom->shape(start_axis + copy_axis);
  }
  if (inferred_axis_ >= 0) {
    int explicit_count = constant_count_;
    explicit_count *= bottom->count(0, start_axis);
    explicit_count *= bottom->count(end_axis);
    for (const auto copy_axis : copy_axes_) {
      explicit_count *= top_shape[start_axis + copy_axis];
    }
    CHECK_EQ(0, (bottom->count() % explicit_count));
    top_shape[start_axis + inferred_axis_] = bottom->count() / explicit_count;
  }
  top->set_shape(top_shape);
  CHECK_EQ(top->count(), bottom->count());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void ReshapeOp::Forward() {
  mutable_tops<float>(0)->share_data(*bottoms<float>(0));
}

REGISTER_OPERATOR(Reshape, ReshapeOp);

}  // namespace Shadow
