#include "pooling_op.hpp"

namespace Shadow {

void PoolingOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int batch = bottom->shape(0), in_c = bottom->shape(1);
  int in_h = bottom->shape(2), in_w = bottom->shape(3);

  if (global_pooling_) {
    kernel_size_h_ = in_h, kernel_size_w_ = in_w;
    stride_h_ = stride_w_ = 1;
    pad_h_ = pad_w_ = 0;
  }

  int out_h =
      pooling_out_size(in_h, kernel_size_h_, stride_h_, pad_h_, full_pooling_);
  int out_w =
      pooling_out_size(in_w, kernel_size_w_, stride_w_, pad_w_, full_pooling_);
  if (pad_h_) {
    if ((out_h - 1) * stride_h_ >= in_h + pad_h_) out_h--;
  }
  if (pad_w_) {
    if ((out_w - 1) * stride_w_ >= in_w + pad_w_) out_w--;
  }

  auto top_shape = bottom->shape();
  top_shape[2] = out_h;
  top_shape[3] = out_w;
  top->reshape(top_shape);

#if defined(USE_CUDNN)
  cudnn::setPooling2dDesc<float>(&pooling_desc_, pool_type_, kernel_size_h_,
                                 kernel_size_w_, pad_h_, pad_w_, stride_h_,
                                 stride_w_);
  cudnn::setTensor4dDesc<float>(&bottom_desc_, batch, in_c, in_h, in_w);
  cudnn::setTensor4dDesc<float>(&top_desc_, batch, in_c, out_h, out_w);

  CUDNN_CHECK(cudnnPoolingForward(
      cudnnHandle_t(op_ws_->CudnnHandle()), pooling_desc_,
      cudnn::dataType<float>::one, bottom_desc_, bottom->data(),
      cudnn::dataType<float>::zero, top_desc_, top->mutable_data()));

#else
  Vision::Pooling(bottom->data(), bottom->shape(), kernel_size_h_,
                  kernel_size_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                  pool_type_, top->shape(), top->mutable_data());
#endif
}

REGISTER_OPERATOR(Pooling, PoolingOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size_h,
             int kernel_size_w, int stride_h, int stride_w, int pad_h,
             int pad_w, int mode, const VecInt &out_shape, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int kistart = h * stride_h - pad_h, kjstart = w * stride_w - pad_w;
          int kiend = std::min(kistart + kernel_size_h, in_h + pad_h);
          int kjend = std::min(kjstart + kernel_size_w, in_w + pad_w);
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
                      int kernel_size_h, int kernel_size_w, int stride_h,
                      int stride_w, int pad_h, int pad_w, int mode,
                      const VecInt &out_shape, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
