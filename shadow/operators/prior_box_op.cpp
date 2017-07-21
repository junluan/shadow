#include "prior_box_op.hpp"

namespace Shadow {

void PriorBoxOp::Setup() {
  min_sizes_ = get_repeated_argument<float>("min_size");
  CHECK_GT(min_sizes_.size(), 0) << "must provide min_size.";
  for (const auto &min_size : min_sizes_) {
    CHECK_GT(min_size, 0) << "min_size must be positive.";
  }
  flip_ = get_single_argument<bool>("flip", true);
  const auto &aspect_ratios = get_repeated_argument<float>("aspect_ratio");
  aspect_ratios_.clear();
  aspect_ratios_.push_back(1.f);
  for (const auto &ratio : aspect_ratios) {
    bool already_exist = false;
    for (const auto &ar : aspect_ratios_) {
      if (std::abs(ratio - ar) < EPS) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      aspect_ratios_.push_back(ratio);
      if (flip_) {
        aspect_ratios_.push_back(1.f / ratio);
      }
    }
  }
  num_priors_ = static_cast<int>(aspect_ratios_.size() * min_sizes_.size());
  max_sizes_ = get_repeated_argument<float>("max_size");
  if (!max_sizes_.empty()) {
    CHECK_EQ(min_sizes_.size(), max_sizes_.size());
    for (int i = 0; i < max_sizes_.size(); ++i) {
      CHECK_GT(max_sizes_[i], min_sizes_[i])
          << "max_size must be greater than min_size.";
      num_priors_ += 1;
    }
  }
  clip_ = get_single_argument<bool>("clip", false);
  variance_ = get_repeated_argument<float>("variance");
  if (variance_.size() > 1) {
    CHECK_EQ(variance_.size(), 4);
  } else if (variance_.empty()) {
    variance_.push_back(0.1);
  }
  if (has_argument("step")) {
    step_ = get_single_argument<float>("step", 0);
    CHECK_GT(step_, 0);
  } else {
    step_ = 0;
  }
  offset_ = get_single_argument<float>("offset", 0.5);
}

void PriorBoxOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  VecInt top_shape{1, 2, 0};
  top_shape[2] = bottom->shape(2) * bottom->shape(3) * num_priors_ * 4;
  CHECK_GT(top_shape[2], 0);
  top->reshape(top_shape);
  top_data_.resize(top->count());

  is_initial_ = false;

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << Util::format_vector(top->shape(), ",", "(", ")");
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

        if (max_sizes_.size() > 0) {
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

void PriorBoxOp::Release() {
  // DLOG(INFO) << "Free PriorBoxOp!";
}

REGISTER_OPERATOR(PriorBox, PriorBoxOp);

}  // namespace Shadow
