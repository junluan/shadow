#include "bias_op.hpp"

namespace Shadow {

void BiasOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  int start_axis = bias_->num_axes() == 0 ? 0 : axis_;
  CHECK_GE(bottom->num_axes(), start_axis + bias_->num_axes());
  for (int i = 0; i < bias_->num_axes(); ++i) {
    CHECK_EQ(bottom->shape(start_axis + i), bias_->shape(i));
  }
  bias_dim_ = bias_->count();
  inner_dim_ = bottom->count(start_axis + bias_->num_axes());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void BiasOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::Bias(bottom->data(), bottom->count(), bias_->data(), bias_dim_,
               inner_dim_, top->mutable_data());
}

REGISTER_OPERATOR(Bias, BiasOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void Bias(const T *in_data, int count, const T *bias_data, int bias_dim,
          int inner_dim, T *out_data) {
  for (int i = 0; i < count; ++i) {
    int index = (i / inner_dim) % bias_dim;
    out_data[i] = in_data[i] + bias_data[index];
  }
}

template void Bias(const float *in_data, int count, const float *bias_data,
                   int bias_dim, int inner_dim, float *out_data);

#elif defined(USE_CL)
template <typename T>
void Bias(const T *in_data, int count, const T *bias_data, int bias_dim,
          int inner_dim, T *out_data) {
  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Bias"];
  kernel->SetArguments(*in_data, count, *bias_data, bias_dim, inner_dim,
                       *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Bias(const BufferF *in_data, int count, const BufferF *bias_data,
                   int bias_dim, int inner_dim, BufferF *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
