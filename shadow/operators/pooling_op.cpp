#include "pooling_op.hpp"

namespace Shadow {

void PoolingOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int batch = bottom->shape(0), in_c = bottom->shape(1);
  int in_h = bottom->shape(2), in_w = bottom->shape(3);
  int out_h =
      pooling_out_size(in_h, kernel_size_, stride_, pad_, full_pooling_);
  int out_w =
      pooling_out_size(in_w, kernel_size_, stride_, pad_, full_pooling_);
  if (pad_) {
    if ((out_h - 1) * stride_ >= in_h + pad_) out_h--;
    if ((out_w - 1) * stride_ >= in_w + pad_) out_w--;
  }

  VecInt top_shape = bottom->shape();
  top_shape[2] = out_h;
  top_shape[3] = out_w;
  top->reshape(top_shape);

  if (global_pooling_) {
    kernel_size_ = in_h;
    stride_ = 1;
    pad_ = 0;
  }

#if defined(USE_CUDNN)
  cudnn::setPooling2dDesc<float>(&pooling_desc_, pool_type_, kernel_size_,
                                 kernel_size_, pad_, pad_, stride_, stride_);
  cudnn::setTensor4dDesc<float>(&bottom_desc_, batch, in_c, in_h, in_w);
  cudnn::setTensor4dDesc<float>(&top_desc_, batch, in_c, out_h, out_w);

  CUDNN_CHECK(cudnnPoolingForward(Kernel::cudnn_handle_, pooling_desc_,
                                  cudnn::dataType<float>::one, bottom_desc_,
                                  bottom->data(), cudnn::dataType<float>::zero,
                                  top_desc_, top->mutable_data()));

#else
  Vision::Pooling(bottom->data(), bottom->shape(), kernel_size_, stride_, pad_,
                  pool_type_, top->shape(), top->mutable_data());
#endif

  DLOG(INFO) << debug_log();
}

REGISTER_OPERATOR(Pooling, PoolingOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int kistart = h * stride - pad, kjstart = w * stride - pad;
          int kiend = std::min(kistart + kernel_size, in_h + pad);
          int kjend = std::min(kjstart + kernel_size, in_w + pad);
          int pool_size = (kiend - kistart) * (kjend - kjstart);
          kistart = std::max(kistart, 0), kjstart = std::max(kjstart, 0);
          kiend = std::min(kiend, in_h), kjend = std::min(kjend, in_w);
          T max = std::numeric_limits<T>::lowest();
          auto sum = T(0);
          for (int ki = kistart; ki < kiend; ++ki) {
            for (int kj = kjstart; kj < kjend; ++kj) {
              int index = kj + in_w * (ki + in_h * (c + in_c * b));
              T value = in_data[index];
              max = (value > max) ? value : max;
              sum += value;
            }
          }
          int out_index = w + out_w * (h + out_h * (c + in_c * b));
          out_data[out_index] = (mode == 0) ? max : sum / pool_size;
        }
      }
    }
  }
}

template void Pooling(const float *in_data, const VecInt &in_shape,
                      int kernel_size, int stride, int pad, int mode,
                      const VecInt &out_shape, float *out_data);

#elif defined(USE_CL)
template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;

  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Pooling"];
  kernel->SetArguments(*in_data, count, in_c, in_h, in_w, kernel_size, stride,
                       pad, mode, out_h, out_w, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Pooling(const BufferF *in_data, const VecInt &in_shape,
                      int kernel_size, int stride, int pad, int mode,
                      const VecInt &out_shape, BufferF *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
