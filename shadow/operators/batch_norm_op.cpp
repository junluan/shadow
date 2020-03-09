#include "batch_norm_op.hpp"

namespace Shadow {

void BatchNormOp::Forward() {
  if (use_global_stats_) {
    CHECK_GE(bottoms_size(), 3);
  } else {
    CHECK_EQ(bottoms_size(), 1);
  }

  const auto bottom = bottoms(0);
  auto top = tops(0);

  top->reshape(bottom->shape());

  int batch = bottom->shape(0), channel = bottom->shape(1),
      spatial_dim = bottom->count(2);

#if defined(USE_CUDNN)
  if (use_cudnn_) {
    cudnn::setTensor4dDesc<float>(&bottom_top_desc_, batch, channel,
                                  spatial_dim, 1);
    cudnn::setTensor4dDesc<float>(&param_desc_, 1, channel, 1, 1);

    ws_->GrowTempBuffer(4 * channel * sizeof(float));

    auto scale_cudnn = ws_->CreateTempBlob({1, channel}, DataType::kF32);
    auto bias_cudnn = ws_->CreateTempBlob({1, channel}, DataType::kF32);
    auto mean_cudnn = ws_->CreateTempBlob({1, channel}, DataType::kF32);
    auto variance_cudnn = ws_->CreateTempBlob({1, channel}, DataType::kF32);

    Blas::Set(channel, 1, scale_cudnn->mutable_data<float>(), 0, ws_->Ctx());
    Blas::Set(channel, 0, bias_cudnn->mutable_data<float>(), 0, ws_->Ctx());

    const auto *mean_cudnn_ptr = bottoms(1)->data<float>(),
               *variance_cudnn_ptr = bottoms(2)->data<float>();
    if (bottoms_size() == 4) {
      float scale = 1;
      CHECK_EQ(bottoms(3)->count(), 1);
      bottoms(3)->get_data<float>(&scale, 1);
      float scale_factor = scale == 0 ? 0 : 1 / scale;
      Blas::Mul(mean_cudnn->count(), bottoms(1)->data<float>(), 0, scale_factor,
                mean_cudnn->mutable_data<float>(), 0, ws_->Ctx());
      Blas::Mul(variance_cudnn->count(), bottoms(2)->data<float>(), 0,
                scale_factor, variance_cudnn->mutable_data<float>(), 0,
                ws_->Ctx());
      mean_cudnn_ptr = mean_cudnn->data<float>();
      variance_cudnn_ptr = variance_cudnn->data<float>();
    }

    double eps = eps_ > CUDNN_BN_MIN_EPSILON ? eps_ : CUDNN_BN_MIN_EPSILON;
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        cudnnHandle_t(ws_->Ctx()->cudnn_handle()), CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<float>::one, cudnn::dataType<float>::zero,
        bottom_top_desc_, bottom->data<float>(), bottom_top_desc_,
        top->mutable_data<float>(), param_desc_, scale_cudnn->data<float>(),
        bias_cudnn->data<float>(), mean_cudnn_ptr, variance_cudnn_ptr, eps));

    return;
  }
#endif

  if (bottom != top) {
    Blas::BlasScopy(bottom->count(), bottom->data<float>(), 0,
                    top->mutable_data<float>(), 0, ws_->Ctx());
  }

  int temp_count =
      2 * channel + bottom->count() + batch * channel + batch + spatial_dim;
  ws_->GrowTempBuffer(temp_count * sizeof(float));

  auto mean = ws_->CreateTempBlob({1, channel}, DataType::kF32);
  auto variance = ws_->CreateTempBlob({1, channel}, DataType::kF32);
  auto temp = ws_->CreateTempBlob(bottom->shape(), DataType::kF32);
  auto batch_by_channel = ws_->CreateTempBlob({batch, channel}, DataType::kF32);
  auto sum_batch_multiplier = ws_->CreateTempBlob({batch}, DataType::kF32);
  auto sum_spatial_multiplier =
      ws_->CreateTempBlob({1, 1, spatial_dim}, DataType::kF32);

  Blas::Set(batch, 1, sum_batch_multiplier->mutable_data<float>(), 0,
            ws_->Ctx());
  Blas::Set(spatial_dim, 1, sum_spatial_multiplier->mutable_data<float>(), 0,
            ws_->Ctx());

  if (use_global_stats_) {
    float scale = 1;
    if (bottoms_size() == 4) {
      CHECK_EQ(bottoms(3)->count(), 1);
      bottoms(3)->get_data<float>(&scale, 1);
    }
    float scale_factor = scale == 0 ? 0 : 1 / scale;
    Blas::Mul(mean->count(), bottoms(1)->data<float>(), 0, scale_factor,
              mean->mutable_data<float>(), 0, ws_->Ctx());
    Blas::Mul(variance->count(), bottoms(2)->data<float>(), 0, scale_factor,
              variance->mutable_data<float>(), 0, ws_->Ctx());
  }

  if (!use_global_stats_) {
    Blas::BlasSgemv(0, batch * channel, spatial_dim,
                    1.f / (batch * spatial_dim), bottom->data<float>(), 0,
                    sum_spatial_multiplier->data<float>(), 0, 0,
                    batch_by_channel->mutable_data<float>(), 0, ws_->Ctx());
    Blas::BlasSgemv(1, batch, channel, 1, batch_by_channel->data<float>(), 0,
                    sum_batch_multiplier->data<float>(), 0, 0,
                    mean->mutable_data<float>(), 0, ws_->Ctx());
  }
  Blas::BlasSgemm(0, 0, batch, channel, 1, 1,
                  sum_batch_multiplier->data<float>(), 0, mean->data<float>(),
                  0, 0, batch_by_channel->mutable_data<float>(), 0, ws_->Ctx());
  Blas::BlasSgemm(0, 0, batch * channel, spatial_dim, 1, -1,
                  batch_by_channel->data<float>(), 0,
                  sum_spatial_multiplier->data<float>(), 0, 1,
                  top->mutable_data<float>(), 0, ws_->Ctx());

  if (!use_global_stats_) {
    Blas::Pow(top->count(), top->data<float>(), 0, 2,
              temp->mutable_data<float>(), 0, ws_->Ctx());
    Blas::BlasSgemv(0, batch * channel, spatial_dim,
                    1.f / (batch * spatial_dim), temp->data<float>(), 0,
                    sum_spatial_multiplier->data<float>(), 0, 0,
                    batch_by_channel->mutable_data<float>(), 0, ws_->Ctx());
    Blas::BlasSgemv(1, batch, channel, 1, batch_by_channel->data<float>(), 0,
                    sum_batch_multiplier->data<float>(), 0, 0,
                    variance->mutable_data<float>(), 0, ws_->Ctx());
  }
  Blas::Add(variance->count(), variance->data<float>(), 0, eps_,
            variance->mutable_data<float>(), 0, ws_->Ctx());
  Blas::Pow(variance->count(), variance->data<float>(), 0, 0.5,
            variance->mutable_data<float>(), 0, ws_->Ctx());
  Blas::BlasSgemm(0, 0, batch, channel, 1, 1,
                  sum_batch_multiplier->data<float>(), 0,
                  variance->data<float>(), 0, 0,
                  batch_by_channel->mutable_data<float>(), 0, ws_->Ctx());
  Blas::BlasSgemm(0, 0, batch * channel, spatial_dim, 1, 1,
                  batch_by_channel->data<float>(), 0,
                  sum_spatial_multiplier->data<float>(), 0, 0,
                  temp->mutable_data<float>(), 0, ws_->Ctx());
  Blas::Div(top->count(), top->data<float>(), 0, temp->data<float>(), 0,
            top->mutable_data<float>(), 0, ws_->Ctx());
}

REGISTER_OPERATOR(BatchNorm, BatchNormOp);

}  // namespace Shadow
