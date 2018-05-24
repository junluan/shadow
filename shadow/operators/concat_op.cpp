#include "concat_op.hpp"

namespace Shadow {

void ConcatOp::Reshape() {
  auto *top = mutable_tops<float>(0);

  int num_axes = bottoms<float>(0)->num_axes();
  auto top_shape = bottoms<float>(0)->shape();
  for (int i = 1; i < bottoms_size(); ++i) {
    CHECK_EQ(num_axes, bottoms<float>(i)->num_axes())
        << "Bottoms must have the same axes!";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) continue;
      CHECK_EQ(top_shape[j], bottoms<float>(i)->shape(j))
          << "Bottoms must have the same shape, except at concat_axis!";
    }
    top_shape[concat_axis_] += bottoms<float>(i)->shape(concat_axis_);
  }
  top->set_shape(top_shape);
  if (bottoms_size() > 1) {
    top->reshape(top_shape);
  }

  VecString str;
  for (int i = 0; i < bottoms_size(); ++i) {
    const auto *bottom = bottoms<float>(i);
    str.push_back(
        Util::format_vector(bottom->shape(), ",", bottom->name() + "(", ")"));
  }
  DLOG(INFO) << op_name_ << "(" << op_type_
             << "): " << Util::format_vector(str, " + ") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void ConcatOp::Forward() {
  auto *top = mutable_tops<float>(0);

  if (bottoms_size() == 1) {
    top->share_data(*bottoms<float>(0));
    return;
  }
  int offset_concat_axis = 0;
  int top_concat_axis = top->shape(concat_axis_);
  for (int i = 0; i < bottoms_size(); ++i) {
    const auto *bottom = bottoms<float>(i);
    int bottom_concat_axis = bottom->shape(concat_axis_);
    int num_concats = bottom->count(0, concat_axis_);
    int concat_input_size = bottom->count(concat_axis_ + 1);
    Vision::Concat(bottom->data(), bottom->count(), num_concats,
                   concat_input_size, top_concat_axis, bottom_concat_axis,
                   offset_concat_axis, top->mutable_data());
    offset_concat_axis += bottom_concat_axis;
  }
}

REGISTER_OPERATOR(Concat, ConcatOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  for (int n = 0; n < num_concats; ++n) {
    memcpy(out_data + (n * top_concat_axis + offset_concat_axis) * concat_size,
           in_data + n * bottom_concat_axis * concat_size,
           bottom_concat_axis * concat_size * sizeof(T));
  }
}

template void Concat(const float *in_data, int count, int num_concats,
                     int concat_size, int top_concat_axis,
                     int bottom_concat_axis, int offset_concat_axis,
                     float *out_data);

#elif defined(USE_CL)
template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Concat"];
  kernel->SetArguments(*in_data, count, num_concats, concat_size,
                       top_concat_axis, bottom_concat_axis, offset_concat_axis,
                       *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Concat(const BufferF *in_data, int count, int num_concats,
                     int concat_size, int top_concat_axis,
                     int bottom_concat_axis, int offset_concat_axis,
                     BufferF *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
