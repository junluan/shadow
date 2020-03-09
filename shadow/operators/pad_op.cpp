#include "pad_op.hpp"

namespace Shadow {

void PadOp::Forward() {
  const auto bottom = bottoms(0);
  auto top = tops(0);

  CHECK_NE(bottom, top);

  int in_h = bottom->shape(2), in_w = bottom->shape(3);

  int out_h = in_h + paddings_[0] + paddings_[1],
      out_w = in_w + paddings_[2] + paddings_[3];

  auto top_shape = bottom->shape();
  top_shape[2] = out_h;
  top_shape[3] = out_w;
  top->reshape(top_shape);

  if (paddings_[0] == 0 && paddings_[1] == 0 && paddings_[2] == 0 &&
      paddings_[3] == 0) {
    Blas::BlasScopy(bottom->count(), bottom->data<float>(), 0,
                    top->mutable_data<float>(), 0, ws_->Ctx());
  } else {
    Blas::Set(top->count(), value_, top->mutable_data<float>(), 0, ws_->Ctx());
    Vision::Pad(bottom->data<float>(), bottom->shape(), paddings_, top->shape(),
                top->mutable_data<float>(), ws_->Ctx());
  }
}

REGISTER_OPERATOR(Pad, PadOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Pad(const T *in_data, const VecInt &in_shape, const VecInt &paddings,
         const VecInt &out_shape, T *out_data, Context *context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < in_h; ++h) {
        if (h + paddings[0] < 0 || h >= in_h + paddings[1]) continue;
        int copy_w = in_w + std::min(paddings[2], 0) + std::min(paddings[3], 0);
        int in_offset = ((b * channel + c) * in_h + h) * in_w;
        int out_offset = ((b * channel + c) * out_h + h + paddings[0]) * out_w;
        if (paddings[2] < 0) {
          in_offset -= paddings[2];
        } else {
          out_offset += paddings[2];
        }
        memcpy(out_data + out_offset, in_data + in_offset, copy_w * sizeof(T));
      }
    }
  }
}

template void Pad(const float *, const VecInt &, const VecInt &, const VecInt &,
                  float *, Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
