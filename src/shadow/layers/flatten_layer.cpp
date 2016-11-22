#include "shadow/layers/flatten_layer.hpp"

void FlattenLayer::Setup(VecBlob *blobs) {
  Layer::Setup(blobs);

  CHECK_NE(tops_[0], bottoms_[0]);

  const auto &flatten_param = layer_param_.flatten_param();

  axis_ = flatten_param.axis();
  end_axis_ = flatten_param.end_axis();
  int num_axes = bottoms_[0]->num_axes();
  if (end_axis_ == -1) end_axis_ = num_axes - 1;
  CHECK_GE(axis_, 0);
  CHECK_LT(end_axis_, num_axes);
  CHECK_LE(axis_, end_axis_);
}

void FlattenLayer::Reshape() {
  for (int i = 0; i < axis_; ++i) {
    tops_[0]->add_shape(bottoms_[0]->shape(i));
  }
  tops_[0]->add_shape(bottoms_[0]->count(axis_, end_axis_ + 1));
  for (int i = end_axis_ + 1; i < bottoms_[0]->num_axes(); ++i) {
    tops_[0]->add_shape(bottoms_[0]->shape(i));
  }

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void FlattenLayer::Forward() { tops_[0]->share_data(bottoms_[0]->data()); }

void FlattenLayer::Release() {
  bottoms_.clear();
  tops_.clear();

  // DLOG(INFO) << "Free FlattenLayer!";
}
