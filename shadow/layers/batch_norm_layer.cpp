#include "batch_norm_layer.hpp"
#include "core/blas.hpp"

namespace Shadow {

void BatchNormLayer::Setup(VecBlobF *blobs) {
  Layer::Setup(blobs);

  const auto &batch_norm_param = layer_param_.batch_norm_param();

  use_global_stats_ = batch_norm_param.use_global_stats();
  if (bottoms_[0]->num_axes() == 1) {
    channels_ = 1;
  } else {
    channels_ = bottoms_[0]->shape(1);
  }

  if (use_global_stats_ && blobs_.size() == 0) {
    for (int i = 0; i < 3; ++i) {
      blobs_.push_back(new BlobF());
    }
    blobs_[0]->reshape(1, channels_);
    blobs_[1]->reshape(1, channels_);
    blobs_[2]->reshape(1);
    Blas::Set(blobs_[0]->count(), 0, blobs_[0]->mutable_data(), 0);
    Blas::Set(blobs_[1]->count(), 1, blobs_[1]->mutable_data(), 0);
    Blas::Set(blobs_[2]->count(), 1, blobs_[2]->mutable_data(), 0);
    DLOG(WARNING) << "Mean, variance and scale params are initialized with the "
                     "default values 0, 1 and 1";
  }

  if (use_global_stats_) {
    CHECK_EQ(blobs_.size(), 3);
    CHECK_EQ(blobs_[2]->count(), 1);
    blobs_[2]->read_data(&scale_);
  }
}

void BatchNormLayer::Reshape() {
  int batch = bottoms_[0]->shape(0), in_h = bottoms_[0]->shape(2),
      in_w = bottoms_[0]->shape(3);

  tops_[0]->reshape(bottoms_[0]->shape());

  spatial_dim_ = in_h * in_w;

  mean_.reshape(1, channels_);
  variance_.reshape(1, channels_);
  temp_.reshape(bottoms_[0]->shape());
  batch_by_channel_.reshape(batch, channels_);

  sum_batch_multiplier_.reshape(batch);
  Blas::Set(batch, 1, sum_batch_multiplier_.mutable_data(), 0);

  sum_spatial_multiplier_.reshape(1, 1, in_h, in_w);
  Blas::Set(spatial_dim_, 1, sum_spatial_multiplier_.mutable_data(), 0);

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void BatchNormLayer::Forward() {
  int batch = bottoms_[0]->shape(0);

  if (bottoms_[0] != tops_[0]) {
    Blas::BlasScopy(bottoms_[0]->count(), bottoms_[0]->data(), 0,
                    tops_[0]->mutable_data(), 0);
  }

  if (use_global_stats_) {
    float scale_factor = scale_ == 0 ? 0 : 1 / scale_;
    Blas::Scale(mean_.count(), scale_factor, blobs_[0]->data(), 0,
                mean_.mutable_data(), 0);
    Blas::Scale(variance_.count(), scale_factor, blobs_[1]->data(), 0,
                variance_.mutable_data(), 0);
  }

  if (!use_global_stats_) {
    Blas::BlasSgemv(0, batch * channels_, spatial_dim_,
                    1.f / (batch * spatial_dim_), bottoms_[0]->data(), 0,
                    sum_spatial_multiplier_.data(), 0, 0,
                    batch_by_channel_.mutable_data(), 0);
    Blas::BlasSgemv(1, batch, channels_, 1, batch_by_channel_.data(), 0,
                    sum_batch_multiplier_.data(), 0, 0, mean_.mutable_data(),
                    0);
  }
  Blas::BlasSgemm(0, 0, batch, channels_, 1, 1, sum_batch_multiplier_.data(), 0,
                  mean_.data(), 0, 0, batch_by_channel_.mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch * channels_, spatial_dim_, 1, -1,
                  batch_by_channel_.data(), 0, sum_spatial_multiplier_.data(),
                  0, 1, tops_[0]->mutable_data(), 0);

  if (!use_global_stats_) {
    Blas::Pow(tops_[0]->count(), tops_[0]->data(), 0, 2, temp_.mutable_data(),
              0);
    Blas::BlasSgemv(0, batch * channels_, spatial_dim_,
                    1.f / (batch * spatial_dim_), temp_.data(), 0,
                    sum_spatial_multiplier_.data(), 0, 0,
                    batch_by_channel_.mutable_data(), 0);
    Blas::BlasSgemv(1, batch, channels_, 1, batch_by_channel_.data(), 0,
                    sum_batch_multiplier_.data(), 0, 0,
                    variance_.mutable_data(), 0);
  }
  Blas::Add(variance_.count(), EPS, variance_.mutable_data(), 0);
  Blas::Pow(variance_.count(), variance_.data(), 0, 0.5,
            variance_.mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch, channels_, 1, 1, sum_batch_multiplier_.data(), 0,
                  variance_.data(), 0, 0, batch_by_channel_.mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch * channels_, spatial_dim_, 1, 1,
                  batch_by_channel_.data(), 0, sum_spatial_multiplier_.data(),
                  0, 0, temp_.mutable_data(), 0);
  Blas::Div(tops_[0]->count(), tops_[0]->data(), 0, temp_.data(), 0,
            tops_[0]->mutable_data(), 0);
}

void BatchNormLayer::Release() {
  mean_.clear(), variance_.clear(), temp_.clear();
  sum_batch_multiplier_.clear();
  sum_spatial_multiplier_.clear();
  batch_by_channel_.clear();

  // DLOG(INFO) << "Free BatchNormLayer!";
}

}  // namespace Shadow
