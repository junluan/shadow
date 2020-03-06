#include "concat_op.hpp"

namespace Shadow {

void ConcatOp::Forward() {
  CHECK_GE(bottoms_size(), 2);

  const auto *bottom_0 = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  axis_ = bottom_0->canonical_index(axis_);

  auto top_shape = bottom_0->shape();
  int num_axes = bottom_0->num_axes();
  for (int n = 1; n < bottoms_size(); ++n) {
    const auto *bottom = bottoms<float>(n);
    CHECK_EQ(num_axes, bottom->num_axes())
        << "Bottoms must have the same axes!";
    for (int d = 0; d < num_axes; ++d) {
      if (d != axis_) {
        CHECK_EQ(top_shape[d], bottom->shape(d))
            << "Bottoms must have the same shape, except at concat_axis!";
      }
    }
    top_shape[axis_] += bottom->shape(axis_);
  }

  top->reshape(top_shape);

  int offset_concat_axis = 0;
  int num_concats = bottom_0->count(0, axis_);
  int concat_size = bottom_0->count(axis_ + 1);
  int top_concat_axis = top->shape(axis_);
  for (int n = 0; n < bottoms_size(); ++n) {
    const auto *bottom = bottoms<float>(n);
    int bottom_concat_axis = bottom->shape(axis_);
    Vision::Concat(bottom->data(), bottom->count(), num_concats, concat_size,
                   top_concat_axis, bottom_concat_axis, offset_concat_axis,
                   top->mutable_data(), op_ws_->Ctx());
    offset_concat_axis += bottom_concat_axis;
  }
}

REGISTER_OPERATOR(Concat, ConcatOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data, Context *context) {
  for (int n = 0; n < num_concats; ++n) {
    memcpy(out_data + (n * top_concat_axis + offset_concat_axis) * concat_size,
           in_data + n * bottom_concat_axis * concat_size,
           bottom_concat_axis * concat_size * sizeof(T));
  }
}

template void Concat(const float *, int, int, int, int, int, int, float *,
                     Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
