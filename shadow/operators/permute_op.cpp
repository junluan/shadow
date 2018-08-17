#include "permute_op.hpp"

namespace Shadow {

void PermuteOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int num_axes = static_cast<int>(permute_order_data_.size());
  CHECK_EQ(num_axes, bottom->num_axes());

  VecInt top_shape, old_steps(num_axes), new_steps(num_axes);
  for (const auto &order : permute_order_data_) {
    top_shape.push_back(bottom->shape(order));
  }
  top->reshape(top_shape);

  for (int d = 0; d < num_axes; ++d) {
    if (d == num_axes - 1) {
      old_steps[d] = 1;
      new_steps[d] = 1;
    } else {
      old_steps[d] = bottom->count(d + 1);
      new_steps[d] = top->count(d + 1);
    }
  }

  permute_order_ = op_ws_->CreateBlob<int>(op_name_ + "_permute_order");
  old_steps_ = op_ws_->CreateBlob<int>(op_name_ + "_old_steps");
  new_steps_ = op_ws_->CreateBlob<int>(op_name_ + "_new_steps");

  permute_order_->reshape({num_axes});
  old_steps_->reshape({num_axes});
  new_steps_->reshape({num_axes});

  permute_order_->set_data(permute_order_data_.data(), num_axes);
  old_steps_->set_data(old_steps.data(), num_axes);
  new_steps_->set_data(new_steps.data(), num_axes);

  Vision::Permute(bottom->data(), bottom->count(), bottom->num_axes(),
                  permute_order_->data(), old_steps_->data(),
                  new_steps_->data(), top->mutable_data());

  DLOG(INFO) << debug_log();
}

REGISTER_OPERATOR(Permute, PermuteOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
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

template void Permute(const float *in_data, int count, int num_axes,
                      const int *permute_order, const int *old_steps,
                      const int *new_steps, float *out_data);

#elif defined(USE_CL)
template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Permute"];
  kernel->SetArguments(*in_data, count, num_axes, *permute_order, *old_steps,
                       *new_steps, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Permute(const BufferF *in_data, int count, int num_axes,
                      const BufferI *permute_order, const BufferI *old_steps,
                      const BufferI *new_steps, BufferF *out_data);
#endif
}  // namespace Vision

}  // namespace Shadow
