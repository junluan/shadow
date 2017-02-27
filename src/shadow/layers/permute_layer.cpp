#include "shadow/layers/permute_layer.hpp"
#include "shadow/util/image.hpp"

void PermuteLayer::Setup(VecBlobF *blobs) {
  Layer::Setup(blobs);

  const auto &permute_param = layer_param_.permute_param();

  num_axes_ = permute_param.order_size();
  CHECK_EQ(num_axes_, bottoms_[0]->num_axes());

  permute_order_data_.clear();
  for (const auto &order : permute_param.order()) {
    permute_order_data_.push_back(order);
  }
}

void PermuteLayer::Reshape() {
  VecInt top_shape, old_steps(num_axes_), new_steps(num_axes_);
  for (const auto &order : permute_order_data_) {
    top_shape.push_back(bottoms_[0]->shape(order));
  }
  tops_[0]->reshape(top_shape);

  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      old_steps[i] = 1;
      new_steps[i] = 1;
    } else {
      old_steps[i] = bottoms_[0]->count(i + 1);
      new_steps[i] = tops_[0]->count(i + 1);
    }
  }

  permute_order_.reshape(num_axes_);
  old_steps_.reshape(num_axes_);
  new_steps_.reshape(num_axes_);

  permute_order_.set_data(permute_order_data_.data());
  old_steps_.set_data(old_steps.data());
  new_steps_.set_data(new_steps.data());

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void PermuteLayer::Forward() {
  Image::Permute(bottoms_[0]->data(), bottoms_[0]->count(),
                 bottoms_[0]->num_axes(), permute_order_.data(),
                 old_steps_.data(), new_steps_.data(),
                 tops_[0]->mutable_data());
}

void PermuteLayer::Release() {
  permute_order_.clear();
  old_steps_.clear();
  new_steps_.clear();

  // DLOG(INFO) << "Free PermuteLayer!";
}
