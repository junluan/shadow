#include "shadow/layers/normalize_layer.hpp"
#include "shadow/util/blas.hpp"

void NormalizeLayer::Setup(VecBlob *blobs) {
  Layer::Setup(blobs);

  const auto &normalize_param = layer_param_.normalize_param();

  across_spatial_ = normalize_param.across_spatial();
  channel_shared_ = normalize_param.channel_shared();

  if (blobs_.size() == 0) {
    if (channel_shared_) {
      scale_.reshape(1);
    } else {
      scale_.reshape(bottoms_[0]->shape(1));
    }
    Blas::Set(scale_.count(), 1, scale_.mutable_data(), 0);
  } else {
    scale_.set_shape(blobs_[0]->shape());
    scale_.share_data(blobs_[0]->mutable_data());
  }

  if (channel_shared_) {
    CHECK_EQ(scale_.count(), 1);
    scale_val_.resize(1);
    scale_.read_data(scale_val_.data());
  } else {
    CHECK_EQ(scale_.count(), bottoms_[0]->shape(1));
  }
}

void NormalizeLayer::Reshape() {
  int in_c = bottoms_[0]->shape(1), in_h = bottoms_[0]->shape(2),
      in_w = bottoms_[0]->shape(3);

  tops_[0]->reshape(bottoms_[0]->shape());

  spatial_dim_ = in_h * in_w;

  if (!across_spatial_) {
    norm_.reshape(1, 1, in_h, in_w);
  }

  buffer_.reshape(1, in_c, in_h, in_w);

  sum_channel_multiplier_.reshape(1, in_c, 1, 1);
  Blas::Set(in_c, 1, sum_channel_multiplier_.mutable_data(), 0);

  sum_spatial_multiplier_.reshape(1, 1, in_h, in_w);
  Blas::Set(spatial_dim_, 1, sum_spatial_multiplier_.mutable_data(), 0);

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void NormalizeLayer::Forward() {
  int batch = bottoms_[0]->shape(0), channels = bottoms_[0]->shape(1);
  int num = bottoms_[0]->num();
  for (int b = 0; b < batch; ++b) {
    int data_offset = b * num;
    Blas::Sqr(num, bottoms_[0]->data(), data_offset, buffer_.mutable_data(), 0);
    if (across_spatial_) {
      float sum = 0;
      Blas::BlasSasum(num, buffer_.data(), 0, &sum);
      float norm = std::sqrt(sum + EPS);
      Blas::Scale(num, 1.f / norm, bottoms_[0]->data(), data_offset,
                  tops_[0]->mutable_data(), data_offset);
    } else {
      Blas::Set(norm_.count(), EPS, norm_.mutable_data(), 0);
      Blas::BlasSgemv(1, channels, spatial_dim_, 1, buffer_.data(), 0,
                      sum_channel_multiplier_.data(), 0, 1,
                      norm_.mutable_data(), 0);
      Blas::Pow(spatial_dim_, norm_.data(), 0, 0.5f, norm_.mutable_data(), 0);
      Blas::BlasSgemm(0, 0, channels, spatial_dim_, 1, 1,
                      sum_channel_multiplier_.data(), 0, norm_.data(), 0, 0,
                      buffer_.mutable_data(), 0);
      Blas::Div(num, bottoms_[0]->data(), data_offset, buffer_.data(), 0,
                tops_[0]->mutable_data(), data_offset);
    }
    if (channel_shared_) {
      Blas::BlasSscal(num, scale_val_[0], tops_[0]->mutable_data(),
                      data_offset);
    } else {
      Blas::BlasSgemm(0, 0, channels, spatial_dim_, 1, 1, scale_.data(), 0,
                      sum_spatial_multiplier_.data(), 0, 0,
                      buffer_.mutable_data(), 0);
      Blas::Mul(num, tops_[0]->data(), data_offset, buffer_.data(), 0,
                tops_[0]->mutable_data(), data_offset);
    }
  }
}

void NormalizeLayer::Release() {
  bottoms_.clear();
  tops_.clear();
  for (auto &blob : blobs_) {
    delete blob;
    blob = nullptr;
  }
  blobs_.clear();

  scale_.clear();
  norm_.clear();
  buffer_.clear();
  sum_channel_multiplier_.clear();
  sum_spatial_multiplier_.clear();

  // DInfo("Free NormalizeLayer!");
}
