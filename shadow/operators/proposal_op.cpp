#include "proposal_op.hpp"

namespace Shadow {

struct RectInfo {
  float xmin, ymin, xmax, ymax, score;
};

inline bool compare_descend(const RectInfo &rect_a, const RectInfo &rect_b) {
  return rect_a.score > rect_b.score;
}

inline float intersection_area(const RectInfo &a, const RectInfo &b) {
  if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax ||
      a.ymax < b.ymin) {
    return 0.f;
  }
  float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
  float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);
  return inter_width * inter_height;
}

inline VecInt nms_sorted(const std::vector<RectInfo> &rects,
                         float nms_threshold) {
  const auto num_rects = rects.size();
  VecFloat areas(num_rects, 0);
  for (int n = 0; n < num_rects; ++n) {
    const auto &rect = rects[n];
    areas[n] = (rect.xmax - rect.xmin + 1) * (rect.ymax - rect.ymin + 1);
  }
  VecInt picked;
  for (int n = 0; n < num_rects; ++n) {
    bool keep = true;
    const auto &rect_n = rects[n];
    for (const auto &p : picked) {
      const auto &rect_p = rects[p];
      float inter_area = intersection_area(rect_n, rect_p);
      float union_area = areas[n] + areas[p] - inter_area;
      if (inter_area / union_area > nms_threshold) {
        keep = false;
      }
    }
    if (keep) {
      picked.push_back(n);
    }
  }
  return picked;
}

void ProposalOp::Reshape() {
  const auto *bottom_score = bottoms<float>(0);
  const auto *bottom_delta = bottoms<float>(1);
  const auto *bottom_info = bottoms<float>(2);
  auto *top = mutable_tops<float>(0);

  int batch = bottom_score->shape(0), in_h = bottom_score->shape(2),
      in_w = bottom_score->shape(3);
  int num_scores = bottom_score->shape(1), num_regs = bottom_delta->shape(1),
      num_info = bottom_info->shape(1);

  CHECK_EQ(batch, 1) << "Only single item batches are supported";
  CHECK_EQ(num_scores, 2 * num_anchors_);
  CHECK_EQ(num_regs, 4 * num_anchors_);
  CHECK_EQ(num_info, 3);

  top->reshape({post_nms_top_n_, 5});

  proposals_ = op_ws_->CreateBlob<float>(op_name_ + "_proposals");
  proposals_->reshape({in_h * in_w * num_anchors_, 6});

  VecString str;
  for (int i = 0; i < bottoms_size(); ++i) {
    const auto *bottom = bottoms<float>(i);
    str.push_back(
        Util::format_vector(bottom->shape(), ",", bottom->name() + "(", ")"));
  }
  DLOG(INFO) << op_name_ << "(" << op_type_
             << "): " << Util::format_vector(str, " + ") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void ProposalOp::Forward() {
  const auto *bottom_score = bottoms<float>(0);
  const auto *bottom_delta = bottoms<float>(1);
  const auto *bottom_info = bottoms<float>(2);
  auto *top = mutable_tops<float>(0);

  Vision::Proposal(anchors_->data(), bottom_score->data(), bottom_delta->data(),
                   bottom_info->data(), bottom_score->shape(), num_anchors_,
                   feat_stride_, min_size_, proposals_->mutable_data());

  const auto *proposal_data = proposals_->cpu_data();

  std::vector<RectInfo> rectangles;
  for (int n = 0; n < proposals_->shape(0); ++n) {
    const auto *proposal_ptr = proposal_data + n * 6;
    if (proposal_ptr[5] > 0) {
      RectInfo rect = {};
      rect.xmin = proposal_ptr[0];
      rect.ymin = proposal_ptr[1];
      rect.xmax = proposal_ptr[2];
      rect.ymax = proposal_ptr[3];
      rect.score = proposal_ptr[4];
      rectangles.push_back(rect);
    }
  }

  std::stable_sort(rectangles.begin(), rectangles.end(), compare_descend);

  if (pre_nms_top_n_ > 0 && pre_nms_top_n_ < rectangles.size()) {
    rectangles.resize(pre_nms_top_n_);
  }

  const auto &picked = nms_sorted(rectangles, nms_thresh_);

  int picked_count = std::min(static_cast<int>(picked.size()), post_nms_top_n_);

  selected_rois_.resize(picked_count * 5, 0);
  for (int n = 0; n < picked_count; ++n) {
    int selected_offset = n * 5;
    const auto &rect = rectangles[picked[n]];
    selected_rois_[selected_offset + 0] = 0;
    selected_rois_[selected_offset + 1] = rect.xmin;
    selected_rois_[selected_offset + 2] = rect.ymin;
    selected_rois_[selected_offset + 3] = rect.xmax;
    selected_rois_[selected_offset + 4] = rect.ymax;
  }

  top->reshape({picked_count, 5});
  top->set_data(selected_rois_.data(), selected_rois_.size());
}

REGISTER_OPERATOR(Proposal, ProposalOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void Proposal(const T *anchor_data, const T *score_data, const T *delta_data,
              const T *info_data, const VecInt &in_shape, int num_anchors,
              int feat_stride, int min_size, T *proposal_data) {
  int in_h = in_shape[2], in_w = in_shape[3], spatial_dim = in_h * in_w;
  int num_proposals = spatial_dim * num_anchors;
  T im_h = info_data[0], im_w = info_data[1], im_scale = info_data[2];
  T min_box_size = min_size * im_scale;
  for (int n = 0; n < num_anchors; ++n) {
    const auto *anchor_ptr = anchor_data + n * 4;
    const auto *score_ptr = score_data + num_proposals + n * spatial_dim;
    const auto *dx_ptr = delta_data + (n * 4 + 0) * spatial_dim;
    const auto *dy_ptr = delta_data + (n * 4 + 1) * spatial_dim;
    const auto *dw_ptr = delta_data + (n * 4 + 2) * spatial_dim;
    const auto *dh_ptr = delta_data + (n * 4 + 3) * spatial_dim;
    T anchor_w = anchor_ptr[2] - anchor_ptr[0] + 1;
    T anchor_h = anchor_ptr[3] - anchor_ptr[1] + 1;
    for (int h = 0; h < in_h; ++h) {
      for (int w = 0; w < in_w; ++w) {
        int spatial_offset = h * in_w + w;
        T anchor_x = anchor_ptr[0] + w * feat_stride;
        T anchor_y = anchor_ptr[1] + h * feat_stride;
        T anchor_cx = anchor_x + (anchor_w - 1) * T(0.5);
        T anchor_cy = anchor_y + (anchor_h - 1) * T(0.5);
        T dx = dx_ptr[spatial_offset], dy = dy_ptr[spatial_offset];
        T dw = dw_ptr[spatial_offset], dh = dh_ptr[spatial_offset];
        T pb_cx = anchor_cx + anchor_w * dx;
        T pb_cy = anchor_cy + anchor_h * dy;
        T pb_w = anchor_w * std::exp(dw), pb_h = anchor_h * std::exp(dh);
        T pb_xmin = pb_cx - (pb_w - 1) * T(0.5);
        T pb_ymin = pb_cy - (pb_h - 1) * T(0.5);
        T pb_xmax = pb_cx + (pb_w - 1) * T(0.5);
        T pb_ymax = pb_cy + (pb_h - 1) * T(0.5);
        auto *prop_ptr = proposal_data + (spatial_offset * num_anchors + n) * 6;
        prop_ptr[0] = std::min(std::max(pb_xmin, T(0)), im_w - 1);
        prop_ptr[1] = std::min(std::max(pb_ymin, T(0)), im_h - 1);
        prop_ptr[2] = std::min(std::max(pb_xmax, T(0)), im_w - 1);
        prop_ptr[3] = std::min(std::max(pb_ymax, T(0)), im_h - 1);
        prop_ptr[4] = score_ptr[spatial_offset];
        pb_w = prop_ptr[2] - prop_ptr[0] + 1;
        pb_h = prop_ptr[3] - prop_ptr[1] + 1;
        prop_ptr[5] = (pb_w >= min_box_size) && (pb_h >= min_box_size);
      }
    }
  }
}

template void Proposal(const float *anchor_data, const float *score_data,
                       const float *delta_data, const float *info_data,
                       const VecInt &in_shape, int num_anchors, int feat_stride,
                       int min_size, float *proposal_data);

#elif defined(USE_CL)
template <typename T>
void Proposal(const T *anchor_data, const T *score_data, const T *delta_data,
              const T *info_data, const VecInt &in_shape, int num_anchors,
              int feat_stride, int min_size, T *proposal_data) {
  int in_h = in_shape[2], in_w = in_shape[3];
  int count = in_h * in_w * num_anchors;

  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["Proposal"];
  kernel->SetArguments(count, *anchor_data, *score_data, *delta_data,
                       *info_data, in_h, in_w, num_anchors, feat_stride,
                       min_size, *proposal_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void Proposal(const BufferF *anchor_data, const BufferF *score_data,
                       const BufferF *delta_data, const BufferF *info_data,
                       const VecInt &in_shape, int num_anchors, int feat_stride,
                       int min_size, BufferF *proposal_data);
#endif

}  // namespace Vision

}  // namespace Shadow
