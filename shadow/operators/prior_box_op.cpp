#include "prior_box_op.hpp"

namespace Shadow {

void PriorBoxOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  VecInt top_shape{1, 2, 0};
  top_shape[2] = bottom->shape(2) * bottom->shape(3) * num_priors_ * 4;
  CHECK_GT(top_shape[2], 0);
  top->reshape(top_shape);
  top_data_.resize(top->count());

  is_initial_ = false;

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void PriorBoxOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (is_initial_) return;

  int height = bottom->shape(2), width = bottom->shape(3);
  int img_height = bottoms<float>(1)->shape(2),
      img_width = bottoms<float>(1)->shape(3);
  float step_h, step_w;
  if (step_ == 0) {
    step_h = static_cast<float>(img_height) / height;
    step_w = static_cast<float>(img_width) / width;
  } else {
    step_h = step_w = step_;
  }
  int idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float center_h = (h + offset_) * step_h;
      float center_w = (w + offset_) * step_w;
      float box_width, box_height;
      for (int s = 0; s < min_sizes_.size(); ++s) {
        float min_size = min_sizes_[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size;
        top_data_[idx++] = (center_w - box_width / 2.f) / img_width;
        top_data_[idx++] = (center_h - box_height / 2.f) / img_height;
        top_data_[idx++] = (center_w + box_width / 2.f) / img_width;
        top_data_[idx++] = (center_h + box_height / 2.f) / img_height;

        if (!max_sizes_.empty()) {
          CHECK_EQ(min_sizes_.size(), max_sizes_.size());
          float max_size = max_sizes_[s];
          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_width = box_height = std::sqrt(min_size * max_size);
          top_data_[idx++] = (center_w - box_width / 2.f) / img_width;
          top_data_[idx++] = (center_h - box_height / 2.f) / img_height;
          top_data_[idx++] = (center_w + box_width / 2.f) / img_width;
          top_data_[idx++] = (center_h + box_height / 2.f) / img_height;
        }

        // rest of priors
        for (const auto &ar : aspect_ratios_) {
          if (std::abs(ar - 1.f) < EPS) {
            continue;
          }
          box_width = min_size * std::sqrt(ar);
          box_height = min_size / std::sqrt(ar);
          top_data_[idx++] = (center_w - box_width / 2.f) / img_width;
          top_data_[idx++] = (center_h - box_height / 2.f) / img_height;
          top_data_[idx++] = (center_w + box_width / 2.f) / img_width;
          top_data_[idx++] = (center_h + box_height / 2.f) / img_height;
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
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
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
}

REGISTER_OPERATOR(PriorBox, PriorBoxOp);

}  // namespace Shadow
