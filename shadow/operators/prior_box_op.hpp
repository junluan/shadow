#ifndef SHADOW_OPERATORS_PRIOR_BOX_OP_HPP
#define SHADOW_OPERATORS_PRIOR_BOX_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PriorBoxOp : public Operator {
 public:
  explicit PriorBoxOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
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

  void Forward() override;

 private:
  int num_priors_;
  float step_, offset_;
  bool flip_, clip_, is_initial_ = false;
  VecFloat min_sizes_, max_sizes_, aspect_ratios_, variance_, top_data_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PRIOR_BOX_OP_HPP
