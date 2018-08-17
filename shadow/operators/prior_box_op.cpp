#include "prior_box_op.hpp"

namespace Shadow {

void PriorBoxOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  const auto *bottom_im = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  if (is_initial_) {
    return;
  }

  int in_h = bottom->shape(2), in_w = bottom->shape(3);
  int im_h = bottom_im->shape(2), im_w = bottom_im->shape(3);

  VecInt top_shape{1, 2, 0};
  top_shape[2] = in_h * in_w * num_priors_ * 4;
  CHECK_GT(top_shape[2], 0);
  top->reshape(top_shape);

  top_data_.resize(top->count());

  float step_h, step_w;
  if (step_ == 0) {
    step_h = static_cast<float>(im_h) / in_h;
    step_w = static_cast<float>(im_w) / in_w;
  } else {
    step_h = step_w = step_;
  }
  int idx = 0;
  for (int h = 0; h < in_h; ++h) {
    for (int w = 0; w < in_w; ++w) {
      float center_h = (h + offset_) * step_h;
      float center_w = (w + offset_) * step_w;
      float box_w, box_h;
      for (int s = 0; s < min_sizes_.size(); ++s) {
        float min_size = min_sizes_[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_w = box_h = min_size;
        top_data_[idx++] = (center_w - box_w / 2.f) / im_w;
        top_data_[idx++] = (center_h - box_h / 2.f) / im_h;
        top_data_[idx++] = (center_w + box_w / 2.f) / im_w;
        top_data_[idx++] = (center_h + box_h / 2.f) / im_h;

        if (!max_sizes_.empty()) {
          CHECK_EQ(min_sizes_.size(), max_sizes_.size());
          float max_size = max_sizes_[s];
          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_w = box_h = std::sqrt(min_size * max_size);
          top_data_[idx++] = (center_w - box_w / 2.f) / im_w;
          top_data_[idx++] = (center_h - box_h / 2.f) / im_h;
          top_data_[idx++] = (center_w + box_w / 2.f) / im_w;
          top_data_[idx++] = (center_h + box_h / 2.f) / im_h;
        }

        // rest of priors
        for (const auto &ar : aspect_ratios_) {
          if (std::abs(ar - 1.f) < EPS) {
            continue;
          }
          box_w = min_size * std::sqrt(ar);
          box_h = min_size / std::sqrt(ar);
          top_data_[idx++] = (center_w - box_w / 2.f) / im_w;
          top_data_[idx++] = (center_h - box_h / 2.f) / im_h;
          top_data_[idx++] = (center_w + box_w / 2.f) / im_w;
          top_data_[idx++] = (center_h + box_h / 2.f) / im_h;
        }
      }
    }
  }
  if (clip_) {
    for (int i = 0; i < top->shape(2); ++i) {
      top_data_[i] = std::min(std::max(top_data_[i], 0.f), 1.f);
    }
  }
  int top_offset = top->shape(2);
  if (variance_.size() == 1) {
    for (int i = 0; i < top->shape(2); ++i) {
      top_data_[top_offset + i] = variance_[0];
    }
  } else {
    int count = 0;
    for (int h = 0; h < in_h; ++h) {
      for (int w = 0; w < in_w; ++w) {
        for (int i = 0; i < num_priors_; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data_[top_offset + (count++)] = variance_[j];
          }
        }
      }
    }
  }
  top->set_data(top_data_.data(), top_data_.size());
  is_initial_ = true;
  DLOG(INFO) << debug_log();
}

REGISTER_OPERATOR(PriorBox, PriorBoxOp);

}  // namespace Shadow
