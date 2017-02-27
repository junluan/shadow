#include "shadow/layers/reorg_layer.hpp"
#include "shadow/util/image.hpp"

void ReorgLayer::Setup(VecBlobF *blobs) {
  Layer::Setup(blobs);

  const auto &reorg_param = layer_param_.reorg_param();

  stride_ = reorg_param.stride();
  CHECK_EQ(bottoms_[0]->shape(2) % stride_, 0);
  CHECK_EQ(bottoms_[0]->shape(3) % stride_, 0);
}

void ReorgLayer::Reshape() {
  int in_c = bottoms_[0]->shape(1);
  int in_h = bottoms_[0]->shape(2), in_w = bottoms_[0]->shape(3);

  VecInt top_shape = bottoms_[0]->shape();
  top_shape[1] = in_c * stride_ * stride_;
  top_shape[2] = in_h / stride_;
  top_shape[3] = in_w / stride_;
  tops_[0]->reshape(top_shape);

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void ReorgLayer::Forward() {
  Image::Reorg(bottoms_[0]->data(), bottoms_[0]->shape(), stride_,
               tops_[0]->mutable_data());
}

void ReorgLayer::Release() {
  // DLOG(INFO) << "Free ReorgLayer!";
}
