#include "scale_op.hpp"

namespace Shadow {

void ScaleOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  int start_axis = scale_->num_axes() == 0 ? 0 : axis_;
  CHECK_GE(bottom->num_axes(), start_axis + scale_->num_axes());
  for (int i = 0; i < scale_->num_axes(); ++i) {
    CHECK_EQ(bottom->shape(start_axis + i), scale_->shape(i));
  }
  scale_dim_ = scale_->count();
  inner_dim_ = bottom->count(start_axis + scale_->num_axes());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void ScaleOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::Scale(bottom->data(), bottom->count(), scale_->data(), bias_->data(),
                scale_dim_, inner_dim_, top->mutable_data());
}

REGISTER_OPERATOR(Scale, ScaleOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data) {
  for (int i = 0; i < count; ++i) {
    int index = (i / inner_dim) % scale_dim;
    out_data[i] = in_data[i] * scale_data[index] + bias_data[index];
  }
}

template void Scale(const float *in_data, int count, const float *scale_data,
                    const float *bias_data, int scale_dim, int inner_dim,
                    float *out_data);

#elif defined(USE_CL)
template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data) {
  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Scale"];
  kernel->SetArguments(*in_data, count, *scale_data, *bias_data, scale_dim,
                       inner_dim, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Scale(const BufferF *in_data, int count,
                    const BufferF *scale_data, const BufferF *bias_data,
                    int scale_dim, int inner_dim, BufferF *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
