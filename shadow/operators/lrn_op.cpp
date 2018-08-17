#include "lrn_op.hpp"

namespace Shadow {

void LRNOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  scale_ = op_ws_->CreateTempBlob<float>(bottom->shape(), op_name_ + "_scale");

  Vision::LRN(bottom->data(), bottom->shape(), size_, alpha_, beta_, k_,
              scale_->mutable_data(), top->mutable_data());

  DLOG(INFO) << debug_log();
}

REGISTER_OPERATOR(LRN, LRNOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void LRN(const T *in_data, const VecInt &in_shape, int size, float alpha,
         float beta, float k, T *scale_data, T *out_data) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int step = in_h * in_w, count = batch * in_c * step;
  int pre_pad = (size - 1) / 2, post_pad = size - pre_pad - 1;
  float alpha_over_size = alpha / size;
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < in_h; ++h) {
      for (int w = 0; w < in_w; ++w) {
        int offset = (b * in_c * in_h + h) * in_w + w, head = 0;
        const T *in_off = in_data + offset;
        T *scale_off = scale_data + offset;
        auto accum_scale = T(0);
        while (head < post_pad && head < in_c) {
          accum_scale += in_off[head * step] * in_off[head * step];
          head++;
        }
        while (head < in_c) {
          accum_scale += in_off[head * step] * in_off[head * step];
          if (head - size >= 0) {
            accum_scale -=
                in_off[(head - size) * step] * in_off[(head - size) * step];
          }
          scale_off[(head - post_pad) * step] =
              k + accum_scale * alpha_over_size;
          head++;
        }
        while (head < in_c + post_pad) {
          if (head - size >= 0) {
            accum_scale -=
                in_off[(head - size) * step] * in_off[(head - size) * step];
          }
          scale_off[(head - post_pad) * step] =
              k + accum_scale * alpha_over_size;
          head++;
        }
      }
    }
  }
  for (int i = 0; i < count; ++i) {
    out_data[i] = in_data[i] * std::pow(scale_data[i], -beta);
  }
}

template void LRN(const float *in_data, const VecInt &in_shape, int size,
                  float alpha, float beta, float k, float *scale_data,
                  float *out_data);

#elif defined(USE_CL)
template <typename T>
void LRN(const T *in_data, const VecInt &in_shape, int size, float alpha,
         float beta, float k, T *scale_data, T *out_data) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  float alpha_over_size = alpha / size, negative_beta = -beta;
  int count = batch * in_h * in_w;

  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["LRNFillScale"];
  kernel->SetArguments(*in_data, count, in_c, in_h, in_w, size, alpha_over_size,
                       k, *scale_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();

  count *= in_c;
  global = count;
  kernel = Kernel::cl_kernels_["LRN"];
  kernel->SetArguments(*in_data, count, *scale_data, negative_beta, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void LRN(const BufferF *in_data, const VecInt &in_shape, int size,
                  float alpha, float beta, float k, BufferF *scale_data,
                  BufferF *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
