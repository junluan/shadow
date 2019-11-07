#include "permute_op.hpp"

namespace Shadow {

void PermuteOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int num_axes = static_cast<int>(permute_order_value_.size());
  CHECK_EQ(num_axes, bottom->num_axes());

  VecInt top_shape, old_steps_value(num_axes), new_steps_value(num_axes);
  for (const auto &order : permute_order_value_) {
    top_shape.push_back(bottom->shape(order));
  }
  top->reshape(top_shape);

  for (int d = 0; d < num_axes; ++d) {
    if (d == num_axes - 1) {
      old_steps_value[d] = 1;
      new_steps_value[d] = 1;
    } else {
      old_steps_value[d] = bottom->count(d + 1);
      new_steps_value[d] = top->count(d + 1);
    }
  }

  op_ws_->GrowTempBuffer(3 * num_axes, sizeof(int));

  auto *permute_order =
      op_ws_->CreateTempBlob<int>({num_axes}, op_name_ + "/permute_order");
  auto *old_steps =
      op_ws_->CreateTempBlob<int>({num_axes}, op_name_ + "/old_steps");
  auto *new_steps =
      op_ws_->CreateTempBlob<int>({num_axes}, op_name_ + "/new_steps");

  permute_order->set_data(permute_order_value_.data(), num_axes);
  old_steps->set_data(old_steps_value.data(), num_axes);
  new_steps->set_data(new_steps_value.data(), num_axes);

  Vision::Permute(bottom->data(), bottom->count(), bottom->num_axes(),
                  permute_order->data(), old_steps->data(), new_steps->data(),
                  top->mutable_data());
}

REGISTER_OPERATOR(Permute, PermuteOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Permute(const T *in_data, int count, int num_axes,
             const int *permute_order, const int *old_steps,
             const int *new_steps, T *out_data) {
  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes; ++j) {
      int order = permute_order[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    out_data[i] = in_data[old_idx];
  }
}

template void Permute(const float *, int, int, const int *, const int *,
                      const int *, float *);
#endif
}  // namespace Vision

}  // namespace Shadow
