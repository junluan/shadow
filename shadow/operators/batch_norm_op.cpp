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

  top->reshape(bottom->shape());

  int batch = bottom->shape(0), channel = bottom->shape(1),
      spatial_dim = bottom->count(2);

#if defined(USE_CUDNN)
  if (use_cudnn_) {
    cudnn::setTensor4dDesc<float>(&bottom_top_desc_, batch, channel,
                                  spatial_dim, 1);
    cudnn::setTensor4dDesc<float>(&param_desc_, 1, channel, 1, 1);

    op_ws_->GrowTempBuffer(4 * channel, sizeof(float));

    auto *scale_cudnn =
        op_ws_->CreateTempBlob<float>({1, channel}, op_name_ + "/scale_cudnn");
    auto *bias_cudnn =
        op_ws_->CreateTempBlob<float>({1, channel}, op_name_ + "/bias_cudnn");
    auto *mean_cudnn =
        op_ws_->CreateTempBlob<float>({1, channel}, op_name_ + "/mean_cudnn");
    auto *variance_cudnn = op_ws_->CreateTempBlob<float>(
        {1, channel}, op_name_ + "/variance_cudnn");

    Blas::Set(channel, 1, scale_cudnn->mutable_data(), 0);
    Blas::Set(channel, 0, bias_cudnn->mutable_data(), 0);

    const auto *mean_cudnn_ptr = bottoms<float>(1)->data(),
               *variance_cudnn_ptr = bottoms<float>(2)->data();
    if (bottoms_size() == 4) {
      float scale = 1;
      CHECK_EQ(bottoms<float>(3)->count(), 1);
      bottoms<float>(3)->get_data(&scale, 1);
      float scale_factor = scale == 0 ? 0 : 1 / scale;
      Blas::Mul(mean_cudnn->count(), bottoms<float>(1)->data(), 0, scale_factor,
                mean_cudnn->mutable_data(), 0);
      Blas::Mul(variance_cudnn->count(), bottoms<float>(2)->data(), 0,
                scale_factor, variance_cudnn->mutable_data(), 0);
      mean_cudnn_ptr = mean_cudnn->data();
      variance_cudnn_ptr = variance_cudnn->data();
    }

    double eps = eps_ > CUDNN_BN_MIN_EPSILON ? eps_ : CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        cudnnHandle_t(op_ws_->Ctx()->cudnn_handle()), CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<float>::one, cudnn::dataType<float>::zero,
        bottom_top_desc_, bottom->data(), bottom_top_desc_, top->mutable_data(),
        param_desc_, scale_cudnn->data(), bias_cudnn->data(), mean_cudnn_ptr,
        variance_cudnn_ptr, eps));

    return;
  }
#endif

  if (bottom != top) {
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0,
                    op_ws_->Ctx()->blas_handle());
  }

  int temp_count =
      2 * channel + bottom->count() + batch * channel + batch + spatial_dim;
  op_ws_->GrowTempBuffer(temp_count, sizeof(float));

  auto *mean = op_ws_->CreateTempBlob<float>({1, channel}, op_name_ + "/mean");
  auto *variance =
      op_ws_->CreateTempBlob<float>({1, channel}, op_name_ + "/variance");
  auto *temp =
      op_ws_->CreateTempBlob<float>(bottom->shape(), op_name_ + "/temp");
  auto *batch_by_channel = op_ws_->CreateTempBlob<float>(
      {batch, channel}, op_name_ + "/batch_by_channel");
  auto *sum_batch_multiplier = op_ws_->CreateTempBlob<float>(
      {batch}, op_name_ + "/sum_batch_multiplier");
  auto *sum_spatial_multiplier = op_ws_->CreateTempBlob<float>(
      {1, 1, spatial_dim}, op_name_ + "/sum_spatial_multiplier");

  Blas::Set(batch, 1, sum_batch_multiplier->mutable_data(), 0);
  Blas::Set(spatial_dim, 1, sum_spatial_multiplier->mutable_data(), 0);

  if (use_global_stats_) {
    float scale = 1;
    if (bottoms_size() == 4) {
      CHECK_EQ(bottoms<float>(3)->count(), 1);
      bottoms<float>(3)->get_data(&scale, 1);
    }
    float scale_factor = scale == 0 ? 0 : 1 / scale;
    Blas::Mul(mean->count(), bottoms<float>(1)->data(), 0, scale_factor,
              mean->mutable_data(), 0);
    Blas::Mul(variance->count(), bottoms<float>(2)->data(), 0, scale_factor,
              variance->mutable_data(), 0);
  }

  if (!use_global_stats_) {
    Blas::BlasSgemv(
        0, batch * channel, spatial_dim, 1.f / (batch * spatial_dim),
        bottom->data(), 0, sum_spatial_multiplier->data(), 0, 0,
        batch_by_channel->mutable_data(), 0, op_ws_->Ctx()->blas_handle());
    Blas::BlasSgemv(1, batch, channel, 1, batch_by_channel->data(), 0,
                    sum_batch_multiplier->data(), 0, 0, mean->mutable_data(), 0,
                    op_ws_->Ctx()->blas_handle());
  }
  Blas::BlasSgemm(0, 0, batch, channel, 1, 1, sum_batch_multiplier->data(), 0,
                  mean->data(), 0, 0, batch_by_channel->mutable_data(), 0,
                  op_ws_->Ctx()->blas_handle());
  Blas::BlasSgemm(0, 0, batch * channel, spatial_dim, 1, -1,
                  batch_by_channel->data(), 0, sum_spatial_multiplier->data(),
                  0, 1, top->mutable_data(), 0, op_ws_->Ctx()->blas_handle());

  if (!use_global_stats_) {
    Blas::Pow(top->count(), top->data(), 0, 2, temp->mutable_data(), 0);
    Blas::BlasSgemv(
        0, batch * channel, spatial_dim, 1.f / (batch * spatial_dim),
        temp->data(), 0, sum_spatial_multiplier->data(), 0, 0,
        batch_by_channel->mutable_data(), 0, op_ws_->Ctx()->blas_handle());
    Blas::BlasSgemv(1, batch, channel, 1, batch_by_channel->data(), 0,
                    sum_batch_multiplier->data(), 0, 0,
                    variance->mutable_data(), 0, op_ws_->Ctx()->blas_handle());
  }
  Blas::Add(variance->count(), variance->data(), 0, eps_,
            variance->mutable_data(), 0);
  Blas::Pow(variance->count(), variance->data(), 0, 0.5,
            variance->mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch, channel, 1, 1, sum_batch_multiplier->data(), 0,
                  variance->data(), 0, 0, batch_by_channel->mutable_data(), 0,
                  op_ws_->Ctx()->blas_handle());
  Blas::BlasSgemm(0, 0, batch * channel, spatial_dim, 1, 1,
                  batch_by_channel->data(), 0, sum_spatial_multiplier->data(),
                  0, 0, temp->mutable_data(), 0, op_ws_->Ctx()->blas_handle());
  Blas::Div(top->count(), top->data(), 0, temp->data(), 0, top->mutable_data(),
            0);
}

REGISTER_OPERATOR(BatchNorm, BatchNormOp);

}  // namespace Shadow
