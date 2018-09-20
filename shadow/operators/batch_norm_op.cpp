#include "batch_norm_op.hpp"

namespace Shadow {

void BatchNormOp::Forward() {
  if (use_global_stats_) {
    CHECK_GE(bottoms_size(), 3);
  } else {
    CHECK_EQ(bottoms_size(), 1);
  }

  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0), channels, spatial_dim = bottom->count(2);

  if (bottom->num_axes() == 1) {
    channels = 1;
  } else {
    channels = bottom->shape(1);
  }

  if (bottom != top) {
    top->reshape(bottom->shape());
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0);
  }

  int temp_count =
      2 * channels + bottom->count() + batch * channels + batch + spatial_dim;
  op_ws_->GrowTempBuffer(temp_count, sizeof(float));

  mean_ = op_ws_->CreateTempBlob<float>({1, channels}, op_name_ + "_mean");
  variance_ =
      op_ws_->CreateTempBlob<float>({1, channels}, op_name_ + "_variance");
  temp_ = op_ws_->CreateTempBlob<float>(bottom->shape(), op_name_ + "_temp");
  batch_by_channel_ = op_ws_->CreateTempBlob<float>(
      {batch, channels}, op_name_ + "_batch_by_channel");
  sum_batch_multiplier_ = op_ws_->CreateTempBlob<float>(
      {batch}, op_name_ + "_sum_batch_multiplier");
  sum_spatial_multiplier_ = op_ws_->CreateTempBlob<float>(
      {1, 1, spatial_dim}, op_name_ + "_sum_spatial_multiplier");

  Blas::Set(batch, 1, sum_batch_multiplier_->mutable_data(), 0);
  Blas::Set(spatial_dim, 1, sum_spatial_multiplier_->mutable_data(), 0);

  if (use_global_stats_) {
    float scale = 1;
    if (bottoms_size() == 4) {
      CHECK_EQ(bottoms<float>(3)->count(), 1);
      bottoms<float>(3)->read_data(&scale, 1);
    }
    float scale_factor = scale == 0 ? 0 : 1 / scale;
    Blas::Mul(mean_->count(), bottoms<float>(1)->data(), 0, scale_factor,
              mean_->mutable_data(), 0);
    Blas::Mul(variance_->count(), bottoms<float>(2)->data(), 0, scale_factor,
              variance_->mutable_data(), 0);
  }

  if (!use_global_stats_) {
    Blas::BlasSgemv(0, batch * channels, spatial_dim,
                    1.f / (batch * spatial_dim), bottom->data(), 0,
                    sum_spatial_multiplier_->data(), 0, 0,
                    batch_by_channel_->mutable_data(), 0);
    Blas::BlasSgemv(1, batch, channels, 1, batch_by_channel_->data(), 0,
                    sum_batch_multiplier_->data(), 0, 0, mean_->mutable_data(),
                    0);
  }
  Blas::BlasSgemm(0, 0, batch, channels, 1, 1, sum_batch_multiplier_->data(), 0,
                  mean_->data(), 0, 0, batch_by_channel_->mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch * channels, spatial_dim, 1, -1,
                  batch_by_channel_->data(), 0, sum_spatial_multiplier_->data(),
                  0, 1, top->mutable_data(), 0);

  if (!use_global_stats_) {
    Blas::Pow(top->count(), top->data(), 0, 2, temp_->mutable_data(), 0);
    Blas::BlasSgemv(0, batch * channels, spatial_dim,
                    1.f / (batch * spatial_dim), temp_->data(), 0,
                    sum_spatial_multiplier_->data(), 0, 0,
                    batch_by_channel_->mutable_data(), 0);
    Blas::BlasSgemv(1, batch, channels, 1, batch_by_channel_->data(), 0,
                    sum_batch_multiplier_->data(), 0, 0,
                    variance_->mutable_data(), 0);
  }
  Blas::Add(variance_->count(), variance_->data(), 0, eps_,
            variance_->mutable_data(), 0);
  Blas::Pow(variance_->count(), variance_->data(), 0, 0.5,
            variance_->mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch, channels, 1, 1, sum_batch_multiplier_->data(), 0,
                  variance_->data(), 0, 0, batch_by_channel_->mutable_data(),
                  0);
  Blas::BlasSgemm(0, 0, batch * channels, spatial_dim, 1, 1,
                  batch_by_channel_->data(), 0, sum_spatial_multiplier_->data(),
                  0, 0, temp_->mutable_data(), 0);
  Blas::Div(top->count(), top->data(), 0, temp_->data(), 0, top->mutable_data(),
            0);
}

REGISTER_OPERATOR(BatchNorm, BatchNormOp);

}  // namespace Shadow
