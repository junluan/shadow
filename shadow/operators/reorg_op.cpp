#include "reorg_op.hpp"

namespace Shadow {

void ReorgOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);

  CHECK_EQ(in_h % stride_, 0);
  CHECK_EQ(in_w % stride_, 0);

  auto top_shape = bottom->shape();
  top_shape[1] = in_c * stride_ * stride_;
  top_shape[2] = in_h / stride_;
  top_shape[3] = in_w / stride_;
  top->reshape(top_shape);

  Vision::Reorg(bottom->data(), bottom->shape(), stride_, top->mutable_data());
}

REGISTER_OPERATOR(Reorg, ReorgOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_c = in_c / (stride * stride);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      for (int h = 0; h < in_h; ++h) {
        for (int w = 0; w < in_w; ++w) {
          int c2 = c % out_c;
          int offset = c / out_c;
          int h2 = h * stride + offset / stride;
          int w2 = w * stride + offset % stride;
          int in_index = ((b * in_c + c) * in_h + h) * in_w + w;
          int out_index =
              ((b * out_c + c2) * in_h * stride + h2) * in_w * stride + w2;
          out_data[in_index] = in_data[out_index];
        }
      }
    }
  }
}

template void Reorg(const float *in_data, const VecInt &in_shape, int stride,
                    float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
