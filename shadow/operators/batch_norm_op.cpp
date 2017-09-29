#include "batch_norm_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void BatchNormOp::Setup() {
  use_global_stats_ = get_single_argument<bool>("use_global_stats", true);
  eps_ = get_single_argument<float>("eps", 1e-5);
  if (bottoms<float>(0)->num_axes() == 1) {
    channels_ = 1;
  } else {
    channels_ = bottoms<float>(0)->shape(1);
  }

  if (use_global_stats_ && blobs_size() == 0) {
    for (int i = 0; i < 3; ++i) {
      add_blobs<float>(op_name_ + "_param_" + Util::to_string(i));
    }
    auto *blob_mean = mutable_blobs<float>(0);
    auto *blob_variance = mutable_blobs<float>(1);
    auto *blob_scale = mutable_blobs<float>(2);
    blob_mean->reshape({1, channels_});
    blob_variance->reshape({1, channels_});
    blob_scale->reshape({1, 1});
    Blas::Set(blob_mean->count(), 0, blob_mean->mutable_data(), 0);
    Blas::Set(blob_variance->count(), 1, blob_variance->mutable_data(), 0);
    Blas::Set(blob_scale->count(), 1, blob_scale->mutable_data(), 0);
    DLOG(WARNING) << "Mean, variance and scale params are initialized with the "
                     "default values 0, 1 and 1";
  }

  if (use_global_stats_) {
    CHECK_EQ(blobs_size(), 3);
    CHECK_EQ(blobs<float>(2)->count(), 1);
  }
}

void BatchNormOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0), in_h = bottom->shape(2),
      in_w = bottom->shape(3);

  top->reshape(bottom->shape());

  spatial_dim_ = in_h * in_w;

  mean_ = op_ws_->CreateBlob<float>(op_name_ + "_mean");
  variance_ = op_ws_->CreateBlob<float>(op_name_ + "_variance");
  temp_ = op_ws_->CreateBlob<float>(op_name_ + "_temp");
  batch_by_channel_ = op_ws_->CreateBlob<float>(op_name_ + "_batch_by_channel");

  mean_->reshape({1, channels_});
  variance_->reshape({1, channels_});
  temp_->reshape(bottom->shape());
  batch_by_channel_->reshape({batch, channels_});

  sum_batch_multiplier_ =
      op_ws_->CreateBlob<float>(op_name_ + "_sum_batch_multiplier");
  sum_batch_multiplier_->reshape({batch});
  Blas::Set(batch, 1, sum_batch_multiplier_->mutable_data(), 0);

  sum_spatial_multiplier_ =
      op_ws_->CreateBlob<float>(op_name_ + "_sum_spatial_multiplier");
  sum_spatial_multiplier_->reshape({1, 1, in_h, in_w});
  Blas::Set(spatial_dim_, 1, sum_spatial_multiplier_->mutable_data(), 0);

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void BatchNormOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0);

  if (bottom != top) {
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0);
  }

  if (use_global_stats_) {
    blobs<float>(2)->read_data(&scale_, 1);
    float scale_factor = scale_ == 0 ? 0 : 1 / scale_;
    Blas::Mul(mean_->count(), blobs<float>(0)->data(), 0, scale_factor,
              mean_->mutable_data(), 0);
    Blas::Mul(variance_->count(), blobs<float>(1)->data(), 0, scale_factor,
              variance_->mutable_data(), 0);
  }

  if (!use_global_stats_) {
    Blas::BlasSgemv(0, batch * channels_, spatial_dim_,
                    1.f / (batch * spatial_dim_), bottom->data(), 0,
                    sum_spatial_multiplier_->data(), 0, 0,
                    batch_by_channel_->mutable_data(), 0);
    Blas::BlasSgemv(1, batch, channels_, 1, batch_by_channel_->data(), 0,
                    sum_batch_multiplier_->data(), 0, 0, mean_->mutable_data(),
                    0);
  }
  Blas::BlasSgemm(0, 0, batch, channels_, 1, 1, sum_batch_multiplier_->data(),
                  0, mean_->data(), 0, 0, batch_by_channel_->mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch * channels_, spatial_dim_, 1, -1,
                  batch_by_channel_->data(), 0, sum_spatial_multiplier_->data(),
                  0, 1, top->mutable_data(), 0);

  if (!use_global_stats_) {
    Blas::Pow(top->count(), top->data(), 0, 2, temp_->mutable_data(), 0);
    Blas::BlasSgemv(0, batch * channels_, spatial_dim_,
                    1.f / (batch * spatial_dim_), temp_->data(), 0,
                    sum_spatial_multiplier_->data(), 0, 0,
                    batch_by_channel_->mutable_data(), 0);
    Blas::BlasSgemv(1, batch, channels_, 1, batch_by_channel_->data(), 0,
                    sum_batch_multiplier_->data(), 0, 0,
                    variance_->mutable_data(), 0);
  }
  Blas::Add(variance_->count(), variance_->data(), 0, eps_,
            variance_->mutable_data(), 0);
  Blas::Pow(variance_->count(), variance_->data(), 0, 0.5,
            variance_->mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch, channels_, 1, 1, sum_batch_multiplier_->data(),
                  0, variance_->data(), 0, 0, batch_by_channel_->mutable_data(),
                  0);
  Blas::BlasSgemm(0, 0, batch * channels_, spatial_dim_, 1, 1,
                  batch_by_channel_->data(), 0, sum_spatial_multiplier_->data(),
                  0, 0, temp_->mutable_data(), 0);
  Blas::Div(top->count(), top->data(), 0, temp_->data(), 0, top->mutable_data(),
            0);
}

void BatchNormOp::Release() {
  // DLOG(INFO) << "Free BatchNormOp!";
}

REGISTER_OPERATOR(BatchNorm, BatchNormOp);

}  // namespace Shadow
