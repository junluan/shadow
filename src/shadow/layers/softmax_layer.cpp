#include "shadow/layers/softmax_layer.hpp"
#include "shadow/util/blas.hpp"

void SoftmaxLayer::Setup(VecBlob *blobs) {
  Layer::Setup(blobs);

  const auto &softmax_param = layer_param_.softmax_param();

  axis_ = bottoms_[0]->CanonicalIndex(softmax_param.axis());
}

void SoftmaxLayer::Reshape() {
  tops_[0]->reshape(bottoms_[0]->shape());

  outer_num_ = bottoms_[0]->count(0, axis_);
  inner_num_ = bottoms_[0]->count(axis_ + 1);

  VecInt scale_dims = bottoms_[0]->shape();
  scale_dims[axis_] = 1;
  scale_.reshape(scale_dims);

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void SoftmaxLayer::Forward() {
  int count = bottoms_[0]->count(), channels = bottoms_[0]->shape(axis_);
  Blas::BlasScopy(count, bottoms_[0]->data(), 0, tops_[0]->mutable_data(), 0);
  Blas::ChannelMax(outer_num_, channels, inner_num_, tops_[0]->data(),
                   scale_.mutable_data());
  Blas::ChannelSub(count, outer_num_, channels, inner_num_, scale_.data(),
                   tops_[0]->mutable_data());
  Blas::Exp(count, tops_[0]->data(), 0, tops_[0]->mutable_data(), 0);
  Blas::ChannelSum(outer_num_, channels, inner_num_, tops_[0]->data(),
                   scale_.mutable_data());
  Blas::ChannelDiv(count, outer_num_, channels, inner_num_, scale_.data(),
                   tops_[0]->mutable_data());
}

void SoftmaxLayer::Release() {
  scale_.clear();

  // DLOG(INFO) << "Free SoftmaxLayer!";
}
