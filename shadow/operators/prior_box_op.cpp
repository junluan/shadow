#include "core/operator.hpp"

namespace Shadow {

class PriorBoxOp : public Operator {
 public:
  PriorBoxOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    min_sizes_ = get_repeated_argument<float>("min_size");
    CHECK_GT(min_sizes_.size(), 0) << "must provide min_size.";
    for (const auto& min_size : min_sizes_) {
      CHECK_GT(min_size, 0) << "min_size must be positive.";
    }
    flip_ = get_single_argument<bool>("flip", true);
    const auto& aspect_ratios = get_repeated_argument<float>("aspect_ratio");
    aspect_ratios_.clear();
    aspect_ratios_.push_back(1.f);
    for (const auto& ratio : aspect_ratios) {
      bool already_exist = false;
      for (const auto& ar : aspect_ratios_) {
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
    step_ = get_single_argument<float>("step", 0);
    offset_ = get_single_argument<float>("offset", 0.5);
  }

  void Run() override {
    const auto bottom = bottoms(0);
    const auto bottom_im = bottoms(1);
    auto top = tops(0);

    if (is_initial_) {
      return;
    }

    int in_h = bottom->shape(2), in_w = bottom->shape(3);
    int im_h = bottom_im->shape(2), im_w = bottom_im->shape(3);

    VecInt top_shape{1, 2, in_h * in_w * num_priors_ * 4};
    CHECK_GT(top_shape[2], 0);
    top->reshape(top_shape);

    top_data_.resize(top->count());

    float step_h, step_w;
    if (std::abs(step_) < EPS) {
      step_h = static_cast<float>(im_h) / in_h;
      step_w = static_cast<float>(im_w) / in_w;
    } else {
      CHECK_GT(step_, 0);
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
          for (const auto& ar : aspect_ratios_) {
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
    int top_offset = top->shape(2);
    if (clip_) {
      for (int i = 0; i < top_offset; ++i) {
        top_data_[i] = std::min(std::max(top_data_[i], 0.f), 1.f);
      }
    }
    if (variance_.size() == 1) {
      for (int i = 0; i < top_offset; ++i) {
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
    top->set_data<float>(top_data_.data(), top_data_.size());
    is_initial_ = true;
  }

 private:
  int num_priors_;
  float step_, offset_;
  bool flip_, clip_, is_initial_ = false;
  VecFloat min_sizes_, max_sizes_, aspect_ratios_, variance_, top_data_;
};

REGISTER_OPERATOR(PriorBox, PriorBoxOp);

}  // namespace Shadow
