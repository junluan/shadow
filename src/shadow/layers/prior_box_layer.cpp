#include "shadow/layers/prior_box_layer.hpp"

void PriorBoxLayer::Setup(VecBlob *blobs) {
  Layer::Setup(blobs);

  const auto &prior_box_param = layer_param_.prior_box_param();

  CHECK(prior_box_param.has_min_size());
  min_size_ = prior_box_param.min_size();
  flip_ = prior_box_param.flip();
  aspect_ratios_.clear();
  aspect_ratios_.push_back(1.f);
  for (const auto &ratio : prior_box_param.aspect_ratio()) {
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
  num_priors_ = aspect_ratios_.size();
  max_size_ = -1;
  if (prior_box_param.has_max_size()) {
    max_size_ = layer_param_.prior_box_param().max_size();
    num_priors_++;
  }
  clip_ = layer_param_.prior_box_param().clip();
  if (prior_box_param.variance_size() > 1) {
    CHECK_EQ(prior_box_param.variance_size(), 4);
    for (const auto &variance : prior_box_param.variance()) {
      variance_.push_back(variance);
    }
  } else if (prior_box_param.variance_size() == 1) {
    variance_.push_back(prior_box_param.variance(0));
  } else {
    variance_.push_back(0.1);
  }
}

void PriorBoxLayer::Reshape() {
  VecInt top_shape(3, 1);
  top_shape[1] = 2;
  top_shape[2] =
      bottoms_[0]->shape(2) * bottoms_[0]->shape(3) * num_priors_ * 4;
  CHECK_GT(top_shape[2], 0);
  tops_[0]->reshape(top_shape);
  top_data_.resize(tops_[0]->count());

  is_initial_ = false;

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void PriorBoxLayer::Forward() {
  if (is_initial_) return;

  int height = bottoms_[0]->shape(2), width = bottoms_[0]->shape(3);
  int img_height = bottoms_[1]->shape(2), img_width = bottoms_[1]->shape(3);
  float step_h = static_cast<float>(img_height) / height;
  float step_w = static_cast<float>(img_width) / width;
  int idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float center_h = (h + 0.5f) * step_h;
      float center_w = (w + 0.5f) * step_w;
      float box_width, box_height;
      // first prior: aspect_ratio = 1, size = min_size
      box_width = box_height = min_size_;
      top_data_[idx++] = (center_w - box_width / 2.f) / img_width;
      top_data_[idx++] = (center_h - box_height / 2.f) / img_height;
      top_data_[idx++] = (center_w + box_width / 2.f) / img_width;
      top_data_[idx++] = (center_h + box_height / 2.f) / img_height;

      if (max_size_ > 0) {
        // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
        box_width = box_height = std::sqrt(min_size_ * max_size_);
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
        box_width = min_size_ * std::sqrt(ar);
        box_height = min_size_ / std::sqrt(ar);
        top_data_[idx++] = (center_w - box_width / 2.f) / img_width;
        top_data_[idx++] = (center_h - box_height / 2.f) / img_height;
        top_data_[idx++] = (center_w + box_width / 2.f) / img_width;
        top_data_[idx++] = (center_h + box_height / 2.f) / img_height;
      }
    }
  }
  if (clip_) {
    for (int i = 0; i < tops_[0]->shape(2); ++i) {
      top_data_[i] = std::min(std::max(top_data_[i], 0.f), 1.f);
    }
  }
  int offset = tops_[0]->shape(2);
  if (variance_.size() == 1) {
    for (int i = 0; i < tops_[0]->shape(2); ++i) {
      top_data_[offset + i] = variance_[0];
    }
  } else {
    int count = 0;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int i = 0; i < num_priors_; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data_[offset + (count++)] = variance_[j];
          }
        }
      }
    }
  }
  tops_[0]->set_data(top_data_.data());
  is_initial_ = true;
}

void PriorBoxLayer::Release() {
  // DLOG(INFO) << "Free PriorBoxLayer!";
}
