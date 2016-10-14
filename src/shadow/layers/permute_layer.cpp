#include "shadow/layers/permute_layer.hpp"
#include "shadow/util/image.hpp"

void PermuteLayer::Reshape() {
  num_axes_ = layer_param_.permute_param().order_size();
  if (num_axes_ != bottom_[0]->num_axes()) {
    Fatal("Number of axes mismatch!");
  }

  VecInt top_shape, permute_order, old_steps(num_axes_), new_steps(num_axes_);

  for (int i = 0; i < num_axes_; ++i) {
    permute_order.push_back(layer_param_.permute_param().order(i));
    top_shape.push_back(bottom_[0]->shape(permute_order[i]));
  }
  top_[0]->reshape(top_shape);

  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      old_steps[i] = 1;
      new_steps[i] = 1;
    } else {
      old_steps[i] = bottom_[0]->count(i + 1);
      new_steps[i] = top_[0]->count(i + 1);
    }
  }

  permute_order_ = new Blob<int>(num_axes_, permute_order.data(),
                                 layer_name_ + " permute_order");
  old_steps_ =
      new Blob<int>(num_axes_, old_steps.data(), layer_name_ + " old_steps");
  new_steps_ =
      new Blob<int>(num_axes_, new_steps.data(), layer_name_ + " new_steps");

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void PermuteLayer::Forward() {
  Image::Permute(bottom_[0]->data(), bottom_[0]->count(),
                 bottom_[0]->num_axes(), permute_order_->data(),
                 old_steps_->data(), new_steps_->data(),
                 top_[0]->mutable_data());
}

void PermuteLayer::Release() {
  bottom_.clear();
  top_.clear();

  permute_order_->clear();
  old_steps_->clear();
  new_steps_->clear();

  // DInfo("Free PermuteLayer!");
}
