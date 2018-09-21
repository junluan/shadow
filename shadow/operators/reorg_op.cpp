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
#endif

}  // namespace Vision

}  // namespace Shadow
