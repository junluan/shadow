#include "shadow/layers/normalize_layer.hpp"
#include "shadow/util/blas.hpp"

void NormalizeLayer::Reshape() {
  across_spatial_ = layer_param_.normalize_param().across_spatial();
  channel_shared_ = layer_param_.normalize_param().channel_shared();

  int num_scale = layer_param_.normalize_param().scale_size();
  for (int i = 0; i < num_scale; ++i) {
    scale_val_.push_back(layer_param_.normalize_param().scale(i));
  }

  int in_c = bottom_[0]->shape(1), in_h = bottom_[0]->shape(2),
      in_w = bottom_[0]->shape(3);

  top_[0]->reshape(bottom_[0]->shape());

  spatial_dim_ = in_h * in_w;

  if (!across_spatial_) {
    norm_.reshape(1, 1, in_h, in_w);
  }
  if (!channel_shared_) {
    for (int i = num_scale; i < in_c; ++i) {
      scale_val_.push_back(1.f);
    }
    scale_.reshape(in_c);
    scale_.set_data(scale_val_.data());
  }

  buffer_.reshape(1, in_c, in_h, in_w);

  sum_channel_multiplier_.reshape(1, in_c, 1, 1);
  Blas::Set(in_c, 1, sum_channel_multiplier_.mutable_data(), 0);

  sum_spatial_multiplier_.reshape(1, 1, in_h, in_w);
  Blas::Set(spatial_dim_, 1, sum_spatial_multiplier_.mutable_data(), 0);

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void NormalizeLayer::Forward() {
  int batch = bottom_[0]->shape(0), channels = bottom_[0]->shape(1);
  int num = bottom_[0]->num();
  for (int b = 0; b < batch; ++b) {
    int data_offset = b * num;
    Blas::Square(num, bottom_[0]->data(), data_offset, buffer_.mutable_data(),
                 0);
    if (across_spatial_) {
      float sum = 0;
      Blas::BlasSasum(num, buffer_.data(), 0, &sum);
      float norm = std::sqrt(sum + EPS);
      Blas::Scale(num, 1.f / norm, bottom_[0]->data(), data_offset,
                  top_[0]->mutable_data(), data_offset);
    } else {
      Blas::Set(norm_.count(), EPS, norm_.mutable_data(), 0);
      Blas::BlasSgemv(1, channels, spatial_dim_, 1, buffer_.data(), 0,
                      sum_channel_multiplier_.data(), 0, 1,
                      norm_.mutable_data(), 0);
      Blas::Pow(spatial_dim_, norm_.data(), 0, 0.5f, norm_.mutable_data(), 0);
      Blas::BlasSgemm(0, 0, channels, spatial_dim_, 1, 1,
                      sum_channel_multiplier_.data(), 0, norm_.data(), 0, 0,
                      buffer_.mutable_data(), 0);
      Blas::Div(num, bottom_[0]->data(), data_offset, buffer_.data(), 0,
                top_[0]->mutable_data(), data_offset);
    }
    if (channel_shared_) {
      Blas::BlasSscal(num, scale_val_[0], top_[0]->mutable_data(), data_offset);
    } else {
      Blas::BlasSgemm(0, 0, channels, spatial_dim_, 1, 1, scale_.data(), 0,
                      sum_spatial_multiplier_.data(), 0, 0,
                      buffer_.mutable_data(), 0);
      Blas::Mul(num, top_[0]->data(), data_offset, buffer_.data(), 0,
                top_[0]->mutable_data(), data_offset);
    }
  }
}

void NormalizeLayer::Release() {
  bottom_.clear();
  top_.clear();

  scale_.clear();
  norm_.clear();
  buffer_.clear();
  sum_channel_multiplier_.clear();
  sum_spatial_multiplier_.clear();

  // DInfo("Free NormalizeLayer!");
}
