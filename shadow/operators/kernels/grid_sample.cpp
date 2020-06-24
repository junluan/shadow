#include "grid_sample.hpp"

#include "util/util.hpp"

namespace Shadow {

namespace Vision {

inline void GridSampleNearest(const float* in_data, const float* grid_data,
                              int batch, int channel, int in_h, int in_w,
                              int out_h, int out_w, int padding_mode,
                              float* out_data) {
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int grid_offset = ((b * out_h + h) * out_w + w) * 2;
          float x = grid_data[grid_offset], y = grid_data[grid_offset + 1];

          int src_h = Util::round((y + 1) / 2.f * (in_h - 1));
          int src_w = Util::round((x + 1) / 2.f * (in_w - 1));

          if (padding_mode == 1) {
            src_h = std::min(std::max(src_h, 0), in_h - 1);
            src_w = std::min(std::max(src_w, 0), in_w - 1);
          } else if (padding_mode == 0) {
            if (src_h < 0 || src_w < 0 || src_h > in_h - 1 ||
                src_w > in_w - 1) {
              *out_data++ = 0.f;
              continue;
            }
          }

          int src_index = ((b * channel + c) * in_h + src_h) * in_w + src_w;
          *out_data++ = in_data[src_index];
        }
      }
    }
  }
}

inline void GridSampleBilinear(const float* in_data, const float* grid_data,
                               int batch, int channel, int in_h, int in_w,
                               int out_h, int out_w, int padding_mode,
                               float* out_data) {
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int grid_offset = ((b * out_h + h) * out_w + w) * 2;
          float x = grid_data[grid_offset], y = grid_data[grid_offset + 1];

          float src_h_f = (y + 1) / 2.f * (in_h - 1);
          float src_w_f = (x + 1) / 2.f * (in_w - 1);

          if (padding_mode == 1) {
            src_h_f = std::min(std::max(src_h_f, 0.f), in_h - 1.f);
            src_w_f = std::min(std::max(src_w_f, 0.f), in_w - 1.f);
          } else if (padding_mode == 0) {
            if (src_h_f < 0 || src_w_f < 0 || src_h_f > in_h - 1 ||
                src_w_f > in_w - 1) {
              *out_data++ = 0.f;
              continue;
            }
          }

          int src_h_0 = std::max(static_cast<int>(std::floor(src_h_f)), 0);
          int src_h_1 =
              std::min(static_cast<int>(std::ceil(src_h_f)), in_h - 1);
          int src_w_0 = std::max(static_cast<int>(std::floor(src_w_f)), 0);
          int src_w_1 =
              std::min(static_cast<int>(std::ceil(src_w_f)), in_w - 1);
          float sh = src_h_f - src_h_0, sw = src_w_f - src_w_0;

          int h_offset = (b * channel + c) * in_h;
          int src_index_0 = (h_offset + src_h_0) * in_w + src_w_0;
          int src_index_1 = (h_offset + src_h_1) * in_w + src_w_0;
          int src_index_2 = (h_offset + src_h_0) * in_w + src_w_1;
          int src_index_3 = (h_offset + src_h_1) * in_w + src_w_1;
          *out_data++ = (1 - sh) * (1 - sw) * in_data[src_index_0] +
                        sh * (1 - sw) * in_data[src_index_1] +
                        (1 - sh) * sw * in_data[src_index_2] +
                        sh * sw * in_data[src_index_3];
        }
      }
    }
  }
}

template <>
void GridSample<DeviceType::kCPU, float>(const float* in_data,
                                         const VecInt& in_shape,
                                         const float* grid_data, int mode,
                                         int padding_mode,
                                         const VecInt& out_shape,
                                         float* out_data, Context* context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  if (mode == 0) {
    GridSampleNearest(in_data, grid_data, batch, channel, in_h, in_w, out_h,
                      out_w, padding_mode, out_data);
  } else if (mode == 1) {
    GridSampleBilinear(in_data, grid_data, batch, channel, in_h, in_w, out_h,
                       out_w, padding_mode, out_data);
  } else {
    LOG(FATAL) << "Unsupported grid sample mode: " << mode;
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(GridSampleCPU,
                           GridSampleKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
