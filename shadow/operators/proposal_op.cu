#include "proposal_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelProposal(int count, const T *anchor_data,
                               const T *score_data, const T *delta_data,
                               const T *info_data, int in_h, int in_w,
                               int num_anchors, int feat_stride, int min_size,
                               T *proposal_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n_out = globalid % num_anchors;
    int w_out = (globalid / num_anchors) % in_w;
    int h_out = globalid / num_anchors / in_w;

    int spatial_dim = in_h * in_w;
    int spatial_offset = h_out * in_w + w_out;
    int delta_offset = n_out * 4 * spatial_dim + spatial_offset;
    T min_box_size = min_size * info_data[2];

    anchor_data += n_out * 4;
    proposal_data += globalid * 6;

    T score = score_data[(num_anchors + n_out) * spatial_dim + spatial_offset];

    T anchor_x = anchor_data[0] + w_out * feat_stride;
    T anchor_y = anchor_data[1] + h_out * feat_stride;
    T anchor_w = anchor_data[2] - anchor_data[0] + 1;
    T anchor_h = anchor_data[3] - anchor_data[1] + 1;
    T anchor_cx = anchor_x + (anchor_w - 1) * T(0.5);
    T anchor_cy = anchor_y + (anchor_h - 1) * T(0.5);

    T dx = delta_data[delta_offset];
    T dy = delta_data[delta_offset + spatial_dim];
    T dw = delta_data[delta_offset + spatial_dim * 2];
    T dh = delta_data[delta_offset + spatial_dim * 3];

    T pb_cx = anchor_cx + anchor_w * dx, pb_cy = anchor_cy + anchor_h * dy;
    T pb_w = anchor_w * std::exp(dw), pb_h = anchor_h * std::exp(dh);

    T pb_xmin = pb_cx - (pb_w - 1) * T(0.5);
    T pb_ymin = pb_cy - (pb_h - 1) * T(0.5);
    T pb_xmax = pb_cx + (pb_w - 1) * T(0.5);
    T pb_ymax = pb_cy + (pb_h - 1) * T(0.5);

    proposal_data[0] = min(max(pb_xmin, T(0)), info_data[1] - 1);
    proposal_data[1] = min(max(pb_ymin, T(0)), info_data[0] - 1);
    proposal_data[2] = min(max(pb_xmax, T(0)), info_data[1] - 1);
    proposal_data[3] = min(max(pb_ymax, T(0)), info_data[0] - 1);
    proposal_data[4] = score;
    pb_w = proposal_data[2] - proposal_data[0] + 1;
    pb_h = proposal_data[3] - proposal_data[1] + 1;
    proposal_data[5] = (pb_w >= min_box_size) && (pb_h >= min_box_size);
  }
}

template <typename T>
void Proposal(const T *anchor_data, const T *score_data, const T *delta_data,
              const T *info_data, const VecInt &in_shape, int num_anchors,
              int feat_stride, int min_size, T *proposal_data) {
  int in_h = in_shape[2], in_w = in_shape[3];
  int count = in_h * in_w * num_anchors;
  KernelProposal<T><<<GetBlocks(count), NumThreads>>>(
      count, anchor_data, score_data, delta_data, info_data, in_h, in_w,
      num_anchors, feat_stride, min_size, proposal_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Proposal(const float *anchor_data, const float *score_data,
                       const float *delta_data, const float *info_data,
                       const VecInt &in_shape, int num_anchors, int feat_stride,
                       int min_size, float *proposal_data);
#endif

}  // namespace Vision

}  // namespace Shadow
