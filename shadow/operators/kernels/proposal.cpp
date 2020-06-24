#include "proposal.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

template <>
void Proposal<DeviceType::kCPU, float>(
    const float* anchor_data, const float* score_data, const float* delta_data,
    const float* info_data, const VecInt& in_shape, int num_anchors,
    int feat_stride, int min_size, float* proposal_data, Context* context) {
  int in_h = in_shape[2], in_w = in_shape[3], spatial_dim = in_h * in_w;
  int num_proposals = spatial_dim * num_anchors;
  auto im_h = info_data[0], im_w = info_data[1], im_scale = info_data[2];
  auto min_box_size = min_size * im_scale;
  for (int n = 0; n < num_anchors; ++n) {
    const auto* anchor_ptr = anchor_data + n * 4;
    const auto* score_ptr = score_data + num_proposals + n * spatial_dim;
    const auto* dx_ptr = delta_data + (n * 4 + 0) * spatial_dim;
    const auto* dy_ptr = delta_data + (n * 4 + 1) * spatial_dim;
    const auto* dw_ptr = delta_data + (n * 4 + 2) * spatial_dim;
    const auto* dh_ptr = delta_data + (n * 4 + 3) * spatial_dim;
    auto anchor_w = anchor_ptr[2] - anchor_ptr[0] + 1;
    auto anchor_h = anchor_ptr[3] - anchor_ptr[1] + 1;
    for (int h = 0; h < in_h; ++h) {
      for (int w = 0; w < in_w; ++w) {
        int spatial_offset = h * in_w + w;
        auto anchor_x = anchor_ptr[0] + w * feat_stride;
        auto anchor_y = anchor_ptr[1] + h * feat_stride;
        auto anchor_cx = anchor_x + (anchor_w - 1) * 0.5f;
        auto anchor_cy = anchor_y + (anchor_h - 1) * 0.5f;
        auto dx = dx_ptr[spatial_offset], dy = dy_ptr[spatial_offset];
        auto dw = dw_ptr[spatial_offset], dh = dh_ptr[spatial_offset];
        auto pb_cx = anchor_cx + anchor_w * dx;
        auto pb_cy = anchor_cy + anchor_h * dy;
        auto pb_w = anchor_w * std::exp(dw), pb_h = anchor_h * std::exp(dh);
        auto pb_xmin = pb_cx - (pb_w - 1) * 0.5f;
        auto pb_ymin = pb_cy - (pb_h - 1) * 0.5f;
        auto pb_xmax = pb_cx + (pb_w - 1) * 0.5f;
        auto pb_ymax = pb_cy + (pb_h - 1) * 0.5f;
        auto* prop_ptr = proposal_data + (spatial_offset * num_anchors + n) * 6;
        prop_ptr[0] = std::min(std::max(pb_xmin, 0.f), im_w - 1.f);
        prop_ptr[1] = std::min(std::max(pb_ymin, 0.f), im_h - 1.f);
        prop_ptr[2] = std::min(std::max(pb_xmax, 0.f), im_w - 1.f);
        prop_ptr[3] = std::min(std::max(pb_ymax, 0.f), im_h - 1.f);
        prop_ptr[4] = score_ptr[spatial_offset];
        pb_w = prop_ptr[2] - prop_ptr[0] + 1;
        pb_h = prop_ptr[3] - prop_ptr[1] + 1;
        prop_ptr[5] = (pb_w >= min_box_size) && (pb_h >= min_box_size);
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ProposalCPU,
                           ProposalKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
