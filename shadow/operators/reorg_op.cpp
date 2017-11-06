#include "reorg_op.hpp"

namespace Shadow {

void ReorgOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);

  VecInt top_shape = bottom->shape();
  top_shape[1] = in_c * stride_ * stride_;
  top_shape[2] = in_h / stride_;
  top_shape[3] = in_w / stride_;
  top->reshape(top_shape);

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void ReorgOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::Reorg(bottom->data(), bottom->shape(), stride_, top->mutable_data());
}

REGISTER_OPERATOR(Reorg, ReorgOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_c = in_c * stride * stride;
  int out_h = in_h / stride, out_w = in_w / stride;
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < out_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int c_in = c % in_c;
          int area = c / in_c;
          int h_in = h * stride + area / stride;
          int w_in = w * stride + area % stride;
          int in_index = ((b * in_c + c_in) * in_h + h_in) * in_w + w_in;
          int out_index = ((b * out_c + c) * out_h + h) * out_w + w;
          out_data[out_index] = in_data[in_index];
        }
      }
    }
  }
}

template void Reorg(const float *in_data, const VecInt &in_shape, int stride,
                    float *out_data);

#elif defined(USE_CL)
template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_c = in_c * stride * stride;
  int out_h = in_h / stride, out_w = in_w / stride;
  int count = batch * out_c * out_h * out_w;

  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Reorg"];
  kernel->SetArguments(*in_data, count, in_c, in_h, in_w, out_c, out_h, out_w,
                       stride, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Reorg(const BufferF *in_data, const VecInt &in_shape, int stride,
                    BufferF *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
