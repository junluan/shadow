#include "shadow/layers/concat_layer.hpp"
#include "shadow/util/image.hpp"

void ConcatLayer::Reshape() {
  int num_axes = bottoms_[0]->num_axes();
  concat_axis_ = layer_param_.concat_param().axis();
  if (concat_axis_ < 0 || concat_axis_ >= num_axes) Fatal("Axis out of range!");

  VecInt top_shape = bottoms_[0]->shape();
  num_concats_ = bottoms_[0]->count(0, concat_axis_ - 1);
  concat_input_size_ = bottoms_[0]->count(concat_axis_ + 1);
  for (int i = 1; i < bottoms_.size(); ++i) {
    if (num_axes != bottoms_[i]->num_axes()) {
      Fatal("Bottoms must have the same axes!");
    }
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) continue;
      if (top_shape[j] != bottoms_[i]->shape(j)) {
        Fatal("Bottoms must have the same shape, except at concat_axis!")
      }
    }
    top_shape[concat_axis_] += bottoms_[i]->shape(concat_axis_);
  }
  tops_[0]->set_shape(top_shape);
  if (bottoms_.size() > 1) {
    tops_[0]->reshape(top_shape);
  }

  std::stringstream out;
  VecString str;
  for (int i = 0; i < bottoms_.size(); ++i) {
    str.push_back(Util::format_vector(bottoms_[i]->shape(), ",", "(", ")"));
  }
  out << layer_name_ << ": " << Util::format_vector(str, " + ") << " -> "
      << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void ConcatLayer::Forward() {
  if (bottoms_.size() == 1) {
    tops_[0]->share_data(bottoms_[0]->mutable_data());
    return;
  }
  int offset_concat_axis = 0;
  int top_concat_axis = tops_[0]->shape(concat_axis_);
  for (int i = 0; i < bottoms_.size(); ++i) {
    int bottom_concat_axis = bottoms_[i]->shape(concat_axis_);
    Image::Concat(bottoms_[i]->data(), bottoms_[i]->count(), num_concats_,
                  concat_input_size_, top_concat_axis, bottom_concat_axis,
                  offset_concat_axis, tops_[0]->mutable_data());
    offset_concat_axis += bottom_concat_axis;
  }
}

void ConcatLayer::Release() {
  bottoms_.clear();
  tops_.clear();

  // DInfo("Free ConcatLayer!");
}
