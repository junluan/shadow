#include "proposal.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelProposal(int count, const float* anchor_data,
                               const float* score_data, const float* delta_data,
                               const float* info_data, int in_h, int in_w,
                               int num_anchors, int feat_stride, int min_size,
                               float* proposal_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n_out = globalid % num_anchors;
    int w_out = (globalid / num_anchors) % in_w;
    int h_out = globalid / num_anchors / in_w;

    int spatial_dim = in_h * in_w;
    int spatial_offset = h_out * in_w + w_out;
    int delta_offset = n_out * 4 * spatial_dim + spatial_offset;
    auto min_box_size = min_size * info_data[2];

    anchor_data += n_out * 4;
    proposal_data += globalid * 6;

    auto score =
        score_data[(num_anchors + n_out) * spatial_dim + spatial_offset];

    auto anchor_x = anchor_data[0] + w_out * feat_stride;
    auto anchor_y = anchor_data[1] + h_out * feat_stride;
    auto anchor_w = anchor_data[2] - anchor_data[0] + 1;
    auto anchor_h = anchor_data[3] - anchor_data[1] + 1;
    auto anchor_cx = anchor_x + (anchor_w - 1) * 0.5f;
    auto anchor_cy = anchor_y + (anchor_h - 1) * 0.5f;

    auto dx = delta_data[delta_offset];
    auto dy = delta_data[delta_offset + spatial_dim];
    auto dw = delta_data[delta_offset + spatial_dim * 2];
    auto dh = delta_data[delta_offset + spatial_dim * 3];

    auto pb_cx = anchor_cx + anchor_w * dx, pb_cy = anchor_cy + anchor_h * dy;
    auto pb_w = anchor_w * expf(dw), pb_h = anchor_h * expf(dh);

    auto pb_xmin = pb_cx - (pb_w - 1) * 0.5f;
    auto pb_ymin = pb_cy - (pb_h - 1) * 0.5f;
    auto pb_xmax = pb_cx + (pb_w - 1) * 0.5f;
    auto pb_ymax = pb_cy + (pb_h - 1) * 0.5f;

    proposal_data[0] = fminf(fmaxf(pb_xmin, 0.f), info_data[1] - 1.f);
    proposal_data[1] = fminf(fmaxf(pb_ymin, 0.f), info_data[0] - 1.f);
    proposal_data[2] = fminf(fmaxf(pb_xmax, 0.f), info_data[1] - 1.f);
    proposal_data[3] = fminf(fmaxf(pb_ymax, 0.f), info_data[0] - 1.f);
    proposal_data[4] = score;
    pb_w = proposal_data[2] - proposal_data[0] + 1;
    pb_h = proposal_data[3] - proposal_data[1] + 1;
    proposal_data[5] = (pb_w >= min_box_size) && (pb_h >= min_box_size);
  }
}

template <>
void Proposal<DeviceType::kGPU, float>(
    const float* anchor_data, const float* score_data, const float* delta_data,
    const float* info_data, const VecInt& in_shape, int num_anchors,
    int feat_stride, int min_size, float* proposal_data, Context* context) {
  int in_h = in_shape[2], in_w = in_shape[3];
  int count = in_h * in_w * num_anchors;
  KernelProposal<<<GetBlocks(count), NumThreads, 0,
                   cudaStream_t(context->stream())>>>(
      count, anchor_data, score_data, delta_data, info_data, in_h, in_w,
      num_anchors, feat_stride, min_size, proposal_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ProposalGPU,
                           ProposalKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
