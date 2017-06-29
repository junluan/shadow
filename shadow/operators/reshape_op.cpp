#include "reshape_op.hpp"

namespace Shadow {

void ReshapeOp::Setup() {
  CHECK_NE(tops_[0], bottoms_[0]);

  axis_ = arg_helper_.GetSingleArgument<int>("axis", 0);
  num_axes_ = arg_helper_.GetSingleArgument<int>("num_axes", -1);
  CHECK_GE(num_axes_, -1);
  shape_ = arg_helper_.GetRepeatedArgument<int>("shape");

  inferred_axis_ = -1;
  copy_axes_.clear();
  constant_count_ = 1;
  for (int i = 0; i < shape_.size(); ++i) {
    int top_dim = shape_[i];
    if (top_dim == 0) {
      copy_axes_.push_back(i);
    } else if (top_dim == -1) {
      CHECK_EQ(inferred_axis_, -1);
      inferred_axis_ = i;
    } else {
      constant_count_ *= top_dim;
    }
  }
}

void ReshapeOp::Reshape() {
  int start_axis = (axis_ >= 0) ? axis_ : (bottoms_[0]->num_axes() + axis_ + 1);
  CHECK_GE(start_axis, 0);
  CHECK_LE(start_axis, bottoms_[0]->num_axes());
  int end_axis =
      (num_axes_ == -1) ? bottoms_[0]->num_axes() : (start_axis + num_axes_);
  CHECK_LE(end_axis, bottoms_[0]->num_axes());
  int num_axes_replaced = end_axis - start_axis;
  int num_axes_retained = bottoms_[0]->num_axes() - num_axes_replaced;
  VecInt top_shape(num_axes_retained + shape_.size());
  int top_shape_index = 0;
  for (int i = 0; i < start_axis; ++i) {
    top_shape[top_shape_index++] = bottoms_[0]->shape(i);
  }
  for (const auto &shape_dim : shape_) {
    top_shape[top_shape_index++] = shape_dim;
  }
  for (int i = end_axis; i < bottoms_[0]->num_axes(); ++i) {
    top_shape[top_shape_index++] = bottoms_[0]->shape(i);
  }
  CHECK_EQ(top_shape_index, top_shape.size());
  for (const auto &copy_axis : copy_axes_) {
    CHECK_GT(bottoms_[0]->num_axes(), start_axis + copy_axis);
    top_shape[start_axis + copy_axis] =
        bottoms_[0]->shape(start_axis + copy_axis);
  }
  if (inferred_axis_ >= 0) {
    int explicit_count = constant_count_;
    explicit_count *= bottoms_[0]->count(0, start_axis);
    explicit_count *= bottoms_[0]->count(end_axis);
    for (const auto &copy_axis : copy_axes_) {
      explicit_count *= top_shape[start_axis + copy_axis];
    }
    CHECK_EQ(0, (bottoms_[0]->count() % explicit_count));
    top_shape[start_axis + inferred_axis_] =
        bottoms_[0]->count() / explicit_count;
  }
  tops_[0]->set_shape(top_shape);
  CHECK_EQ(tops_[0]->count(), bottoms_[0]->count());

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void ReshapeOp::Forward() { tops_[0]->share_data(*bottoms_[0]); }

void ReshapeOp::Release() {
  // DLOG(INFO) << "Free ReshapeOp!";
}

REGISTER_OP_CLASS(Reshape);

}  // namespace Shadow
