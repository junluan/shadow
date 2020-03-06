#include "stack_op.hpp"

namespace Shadow {

void StackOp::Forward() {
  const auto *bottom_0 = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int num_axes = bottom_0->num_axes();
  CHECK(axis_ >= -(num_axes + 1) && axis_ < num_axes + 1)
      << "axis out of bound.";
  if (axis_ < 0) {
    axis_ += num_axes + 1;
  }

  auto top_shape = bottom_0->shape();
  top_shape.insert(top_shape.begin() + axis_, bottoms_size());
  top->reshape(top_shape);

  int num_stacks = bottom_0->count(0, axis_);
  int stack_size = bottom_0->count(axis_);
  int top_stack_axis = top->shape(axis_);
  for (int n = 0; n < bottoms_size(); ++n) {
    const auto *bottom = bottoms<float>(n);
    Vision::Stack(bottom->data(), bottom->count(), num_stacks, stack_size,
                  top_stack_axis, n, top->mutable_data(), op_ws_->Ctx());
  }
}

REGISTER_OPERATOR(Stack, StackOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Stack(const T *in_data, int count, int num_stacks, int stack_size,
           int top_stack_axis, int offset_stack_axis, T *out_data,
           Context *context) {
  for (int n = 0; n < num_stacks; ++n) {
    memcpy(out_data + (n * top_stack_axis + offset_stack_axis) * stack_size,
           in_data + n * stack_size, stack_size * sizeof(T));
  }
}

template void Stack(const float *, int, int, int, int, int, float *, Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
