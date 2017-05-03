#include "lrn_layer.hpp"
#include "core/image.hpp"

void LRNLayer::Setup(VecBlobF *blobs) {
  Layer::Setup(blobs);

  const auto &lrn_param = layer_param_.lrn_param();

  size_ = lrn_param.local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  alpha_ = lrn_param.alpha();
  beta_ = lrn_param.beta();
  norm_region_ = lrn_param.norm_region();
  k_ = lrn_param.k();
}

void LRNLayer::Reshape() {
  tops_[0]->reshape(bottoms_[0]->shape());

  scale_.reshape(bottoms_[0]->shape());

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void LRNLayer::Forward() {
  Image::LRN(bottoms_[0]->data(), bottoms_[0]->shape(), size_, alpha_, beta_,
             k_, scale_.mutable_data(), tops_[0]->mutable_data());
}

void LRNLayer::Release() {
  scale_.clear();

  // DLOG(INFO) << "Free LRNLayer!";
}
