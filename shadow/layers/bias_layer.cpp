#include "bias_layer.hpp"
#include "core/blas.hpp"
#include "core/image.hpp"

namespace Shadow {

void BiasLayer::Setup(VecBlobF *blobs) {
  Layer::Setup(blobs);

  const auto &bias_param = layer_param_.bias_param();

  axis_ = bottoms_[0]->canonical_index(bias_param.axis());
  num_axis_ = bias_param.num_axes();
  CHECK_GE(num_axis_, -1);

  if (bottoms_.size() == 1 && blobs_.size() == 0) {
    int end_axis;
    if (num_axis_ == -1) {
      end_axis = bottoms_[0]->num_axes();
    } else {
      end_axis = axis_ + num_axis_;
      CHECK_GE(bottoms_[0]->num_axes(), end_axis);
    }
    VecInt bias_shape;
    for (int i = axis_; i < end_axis; ++i) {
      bias_shape.push_back(bottoms_[0]->shape(i));
    }
    blobs_.push_back(new BlobF(bias_shape));
    Blas::Set(blobs_[0]->count(), 0, blobs_[0]->mutable_data(), 0);
    DLOG(WARNING) << "Bias param is initialized with the default value 0";
  }
  bias_ = bottoms_.size() > 1 ? bottoms_[1] : blobs_[0];
}

void BiasLayer::Reshape() {
  if (bottoms_[0] != tops_[0]) {
    tops_[0]->reshape(bottoms_[0]->shape());
  }

  int start_axis = bias_->num_axes() == 0 ? 0 : axis_;
  CHECK_GE(bottoms_[0]->num_axes(), start_axis + bias_->num_axes());
  for (int i = 0; i < bias_->num_axes(); ++i) {
    CHECK_EQ(bottoms_[0]->shape(start_axis + i), bias_->shape(i));
  }
  bias_dim_ = bias_->count();
  inner_dim_ = bottoms_[0]->count(start_axis + bias_->num_axes());

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void BiasLayer::Forward() {
  Image::Bias(bottoms_[0]->data(), bottoms_[0]->count(), bias_->data(),
              bias_dim_, inner_dim_, tops_[0]->mutable_data());
}

void BiasLayer::Release() {
  // DLOG(INFO) << "Free BiasLayer!";
}

}  // namespace Shadow
