#include "proposal_op.hpp"

namespace Shadow {

struct RectInfo {
  float xmin, ymin, xmax, ymax, score;
};

inline VecFloat generate_anchors(int base_size, const VecFloat &ratios,
                                 const VecFloat &scales) {
  auto num_ratio = static_cast<int>(ratios.size());
  auto num_scale = static_cast<int>(scales.size());
  VecFloat anchors(4 * num_ratio * num_scale, 0);
  float cx = (base_size - 1) * 0.5f, cy = (base_size - 1) * 0.5f;
  for (int i = 0; i < num_ratio; ++i) {
    float ar = ratios[i];
    float r_w = Util::round(std::sqrt(base_size * base_size / ar));
    float r_h = Util::round(r_w * ar);
    for (int j = 0; j < num_scale; ++j) {
      float scale = scales[j];
      float rs_w = r_w * scale, rs_h = r_h * scale;
      float *anchor = anchors.data() + (i * num_scale + j) * 4;
      anchor[0] = cx - (rs_w - 1) * 0.5f;
      anchor[1] = cy - (rs_h - 1) * 0.5f;
      anchor[2] = cx + (rs_w - 1) * 0.5f;
      anchor[3] = cy + (rs_h - 1) * 0.5f;
    }
  }
  return anchors;
}

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

void ProposalOp::Setup() {
  feat_stride_ = get_single_argument<int>("feat_stride", 16);
  pre_nms_topN_ = 6000;
  post_nms_topN_ = 300;
  min_size_ = 16;
  base_size_ = 16;
  nms_thresh_ = 0.7;
  ratios_ = {0.5f, 1.f, 2.f};
  scales_ = {8.f, 16.f, 32.f};
  anchors_ = generate_anchors(base_size_, ratios_, scales_);
  num_anchors_ = static_cast<int>(ratios_.size() * scales_.size());
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

  top->reshape({post_nms_topN_, 5});

  spatial_dim_ = in_h * in_w;
  num_proposals_ = num_anchors_ * spatial_dim_;
  proposals_.resize(num_proposals_ * 4, 0);

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
  auto *bottom_score = mutable_bottoms<float>(0);
  auto *bottom_delta = mutable_bottoms<float>(1);
  auto *bottom_info = mutable_bottoms<float>(2);
  auto *top = mutable_tops<float>(0);

  int in_h = bottom_score->shape(2), in_w = bottom_score->shape(3);

  const auto *score_data = bottom_score->cpu_data();
  const auto *delta_data = bottom_delta->cpu_data();
  const auto *info_data = bottom_info->cpu_data();
  float im_h = info_data[0], im_w = info_data[1], im_scale = info_data[2];

  const auto *anchors_data = anchors_.data();
  auto *prop_data = proposals_.data();

  // generate proposals from bbox deltas and shifted anchors
  for (int n = 0; n < num_anchors_; ++n) {
    const auto *dx_ptr = delta_data + (n * 4 + 0) * spatial_dim_;
    const auto *dy_ptr = delta_data + (n * 4 + 1) * spatial_dim_;
    const auto *dw_ptr = delta_data + (n * 4 + 2) * spatial_dim_;
    const auto *dh_ptr = delta_data + (n * 4 + 3) * spatial_dim_;

    const auto *anchor_ptr = anchors_data + n * 4;

    float anchor_w = anchor_ptr[2] - anchor_ptr[0] + 1;
    float anchor_h = anchor_ptr[3] - anchor_ptr[1] + 1;

    float anchor_y = anchor_ptr[1];
    for (int h = 0; h < in_h; ++h) {
      float anchor_x = anchor_ptr[0];
      for (int w = 0; w < in_w; ++w) {
        auto *prop_ptr = prop_data + ((n * in_h + h) * in_w + w) * 4;

        float cx = anchor_x + anchor_w * 0.5f;
        float cy = anchor_y + anchor_h * 0.5f;

        float dx = dx_ptr[h * in_w + w];
        float dy = dy_ptr[h * in_w + w];
        float dw = dw_ptr[h * in_w + w];
        float dh = dh_ptr[h * in_w + w];

        float pb_cx = cx + anchor_w * dx;
        float pb_cy = cy + anchor_h * dy;
        float pb_w = anchor_w * std::exp(dw);
        float pb_h = anchor_h * std::exp(dh);

        prop_ptr[0] = pb_cx - pb_w * 0.5f;
        prop_ptr[1] = pb_cy - pb_h * 0.5f;
        prop_ptr[2] = pb_cx + pb_w * 0.5f;
        prop_ptr[3] = pb_cy + pb_h * 0.5f;

        anchor_x += feat_stride_;
      }
      anchor_y += feat_stride_;
    }
  }

  // clip predicted boxes to image
  for (int n = 0; n < num_proposals_; ++n) {
    auto *prop_ptr = prop_data + n * 4;
    prop_ptr[0] = std::max(std::min(prop_ptr[0], im_w - 1), 0.f);
    prop_ptr[1] = std::max(std::min(prop_ptr[1], im_h - 1), 0.f);
    prop_ptr[2] = std::max(std::min(prop_ptr[2], im_w - 1), 0.f);
    prop_ptr[3] = std::max(std::min(prop_ptr[3], im_h - 1), 0.f);
  }

  float min_box_size = min_size_ * im_scale;

  std::vector<RectInfo> rects;
  const auto *score_ptr = score_data + num_proposals_;
  for (int n = 0; n < num_proposals_; ++n) {
    int index = n % spatial_dim_;
    auto *prop_ptr = prop_data + n * 4;
    float pb_w = prop_ptr[2] - prop_ptr[0] + 1;
    float pb_h = prop_ptr[3] - prop_ptr[1] + 1;
    if (pb_w >= min_box_size && pb_h >= min_box_size) {
      RectInfo rect = {};
      rect.xmin = prop_ptr[0];
      rect.ymin = prop_ptr[1];
      rect.xmax = prop_ptr[2];
      rect.ymax = prop_ptr[3];
      rect.score = score_ptr[index];
      rects.push_back(rect);
    }
  }

  // sort all (proposal, score) pairs by score from highest to lowest
  std::stable_sort(rects.begin(), rects.end(), compare_descend);

  // take top pre_nms_topN
  if (pre_nms_topN_ > 0 && pre_nms_topN_ < rects.size()) {
    rects.resize(pre_nms_topN_);
  }

  // apply nms with nms_thresh
  const auto &picked = nms_sorted(rects, nms_thresh_);

  // take post_nms_topN_
  int picked_count = std::min(static_cast<int>(picked.size()), post_nms_topN_);

  selected_rois_.resize(picked_count * 5, 0);
  for (int n = 0; n < picked_count; ++n) {
    int selected_offset = n * 5;
    const auto &rect = rects[picked[n]];
    selected_rois_[selected_offset + 0] = 0;
    selected_rois_[selected_offset + 1] = rect.xmin;
    selected_rois_[selected_offset + 2] = rect.ymin;
    selected_rois_[selected_offset + 3] = rect.xmax;
    selected_rois_[selected_offset + 4] = rect.xmax;
  }

  top->reshape({picked_count, 5});
  top->set_data(selected_rois_.data(), selected_rois_.size());
}

void ProposalOp::Release() {
  // DLOG(INFO) << "Free ProposalOp!";
}

REGISTER_OPERATOR(Proposal, ProposalOp);

}  // namespace Shadow
