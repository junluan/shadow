#include "resize_op.hpp"

namespace Shadow {

void ResizeOp::Forward() {
  const auto* bottom = bottoms<float>(0);
  auto* top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int in_h = bottom->shape(2), in_w = bottom->shape(3);
  int out_h = out_h_, out_w = out_w_;

  if (bottoms_size() > 1) {
    const auto& size_type = bottoms_type(1);
    if (size_type == int_id) {
      const auto* size = bottoms<int>(1);
      CHECK_EQ(size->num_axes(), 1);
      CHECK_EQ(size->count(), 2);
      VecInt size_data(2, 0);
      size->get_data(size_data.data(), 2);
      out_h = size_data[0], out_w = size_data[1];
    } else if (size_type == float_id) {
      const auto* size = bottoms<float>(1);
      CHECK_EQ(size->num_axes(), 4);
      out_h = size->shape(2), out_w = size->shape(3);
    }
  }

  if (out_h == 0 || out_w == 0) {
    out_h = static_cast<int>(in_h * scale_h_);
    out_w = static_cast<int>(in_w * scale_w_);
  }

  CHECK_GT(out_h, 0);
  CHECK_GT(out_w, 0);

  auto top_shape = bottom->shape();
  top_shape[2] = out_h;
  top_shape[3] = out_w;
  top->reshape(top_shape);

  if (out_h == in_h && out_w == in_w) {
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0,
                    op_ws_->Ctx());
  } else {
    // Nearest: 0, Bilinear: 1
    Vision::Resize(bottom->data(), bottom->shape(), type_, align_corners_,
                   top->shape(), top->mutable_data(), op_ws_->Ctx());
  }
}

REGISTER_OPERATOR(Resize, ResizeOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
inline void ResizeNearest(const T* in_data, int batch, int channel, int in_h,
                          int in_w, int out_h, int out_w, T* out_data) {
  float fh = static_cast<float>(in_h) / out_h;
  float fw = static_cast<float>(in_w) / out_w;
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < out_h; ++h) {
        int src_h = static_cast<int>(h * fh);
        for (int w = 0; w < out_w; ++w) {
          int src_w = static_cast<int>(w * fw);
          int src_index = ((b * channel + c) * in_h + src_h) * in_w + src_w;
          *out_data++ = in_data[src_index];
        }
      }
    }
  }
}

template <typename T>
inline void ResizeBilinear(const T* in_data, int batch, int channel, int in_h,
                           int in_w, int out_h, int out_w, bool align_corners,
                           T* out_data) {
  float fh = align_corners ? static_cast<float>(in_h - 1) / (out_h - 1)
                           : static_cast<float>(in_h) / out_h;
  float fw = align_corners ? static_cast<float>(in_w - 1) / (out_w - 1)
                           : static_cast<float>(in_w) / out_w;
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      int src_index_off = (b * channel + c) * in_h;
      for (int h = 0; h < out_h; ++h) {
        float src_h_f;
        if (align_corners) {
          src_h_f = h * fh;
        } else {
          src_h_f = (h + 0.5f) * fh - 0.5f;
          src_h_f = src_h_f < 0 ? 0 : src_h_f;
        }
        int src_h_l = static_cast<int>(src_h_f), src_h_h = src_h_l + 1;
        if (src_h_l >= in_h - 1) {
          src_h_h = src_h_l = in_h - 1;
        }
        float sh = src_h_f - src_h_l;
        for (int w = 0; w < out_w; ++w) {
          float src_w_f;
          if (align_corners) {
            src_w_f = w * fw;
          } else {
            src_w_f = (w + 0.5f) * fw - 0.5f;
            src_w_f = src_w_f < 0 ? 0 : src_w_f;
          }
          int src_w_l = static_cast<int>(src_w_f), src_w_h = src_w_l + 1;
          if (src_w_l >= in_w - 1) {
            src_w_h = src_w_l = in_w - 1;
          }
          float sw = src_w_f - src_w_l;
          int src_index_0 = (src_index_off + src_h_l) * in_w + src_w_l;
          int src_index_1 = (src_index_off + src_h_h) * in_w + src_w_l;
          int src_index_2 = (src_index_off + src_h_l) * in_w + src_w_h;
          int src_index_3 = (src_index_off + src_h_h) * in_w + src_w_h;
          *out_data++ =
              static_cast<T>((1 - sh) * (1 - sw) * in_data[src_index_0] +
                             sh * (1 - sw) * in_data[src_index_1] +
                             (1 - sh) * sw * in_data[src_index_2] +
                             sh * sw * in_data[src_index_3]);
        }
      }
    }
  }
}

template <typename T>
void Resize(const T* in_data, const VecInt& in_shape, int type,
            bool align_corners, const VecInt& out_shape, T* out_data,
            Context* context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  if (type == 0) {
    ResizeNearest(in_data, batch, channel, in_h, in_w, out_h, out_w, out_data);
  } else if (type == 1) {
    ResizeBilinear(in_data, batch, channel, in_h, in_w, out_h, out_w,
                   align_corners, out_data);
  } else {
    LOG(FATAL) << "Unsupported resize type: " << type;
  }
}

template void Resize(const float*, const VecInt&, int, bool, const VecInt&,
                     float*, Context*);
#endif

}  // namespace Vision

}  // namespace Shadow
