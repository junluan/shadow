#include "instance_norm_op.hpp"

#include "scale_op.hpp"

namespace Shadow {

void InstanceNormOp::Forward() {
  CHECK(bottoms_size() == 1 || bottoms_size() == 3);

  const auto bottom = bottoms(0);
  auto top = tops(0);

  top->reshape(bottom->shape());

  int batch = bottom->shape(0), channel = bottom->shape(1),
      spatial_dim = bottom->count(2);

#if defined(USE_CUDNN)
  cudnn::setTensor4dDesc<float>(&bottom_top_desc_, 1, batch * channel,
                                spatial_dim, 1);
  cudnn::setTensor4dDesc<float>(&param_desc_, 1, batch * channel, 1, 1);

  ws_->GrowTempBuffer(2 * batch * channel * sizeof(float));

  auto scale_cudnn = ws_->CreateTempBlob({1, batch * channel}, DataType::kF32);
  auto bias_cudnn = ws_->CreateTempBlob({1, batch * channel}, DataType::kF32);

  if (bottoms_size() == 1) {
    Blas::Set(batch * channel, 1, scale_cudnn->mutable_data<float>(), 0,
              ws_->Ctx());
    Blas::Set(batch * channel, 0, bias_cudnn->mutable_data<float>(), 0,
              ws_->Ctx());
  } else {
    const auto scale = bottoms(1);
    const auto bias = bottoms(2);
    CHECK_EQ(scale->count(), channel);
    CHECK_EQ(bias->count(), channel);
    for (int b = 0; b < batch; ++b) {
      Blas::BlasScopy(channel, scale->data<float>(), 0,
                      scale_cudnn->mutable_data<float>(), b * channel,
                      ws_->Ctx());
      Blas::BlasScopy(channel, bias->data<float>(), 0,
                      bias_cudnn->mutable_data<float>(), b * channel,
                      ws_->Ctx());
    }
  }

  double eps = eps_ > CUDNN_BN_MIN_EPSILON ? eps_ : CUDNN_BN_MIN_EPSILON;
  CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      cudnnHandle_t(ws_->Ctx()->cudnn_handle()), CUDNN_BATCHNORM_SPATIAL,
      cudnn::dataType<float>::one, cudnn::dataType<float>::zero,
      bottom_top_desc_, bottom->data<float>(), bottom_top_desc_,
      top->mutable_data<float>(), param_desc_, scale_cudnn->data<float>(),
      bias_cudnn->data<float>(), 1., nullptr, nullptr, eps, nullptr, nullptr));

#else
  if (bottom != top) {
    Blas::BlasScopy(bottom->count(), bottom->data<float>(), 0,
                    top->mutable_data<float>(), 0, ws_->Ctx());
  }

  int temp_count = batch * channel + bottom->count() + spatial_dim;
  ws_->GrowTempBuffer(temp_count * sizeof(float));

  auto stats = ws_->CreateTempBlob({batch, channel}, DataType::kF32);
  auto temp = ws_->CreateTempBlob(bottom->shape(), DataType::kF32);
  auto sum_spatial_multiplier =
      ws_->CreateTempBlob({1, 1, spatial_dim}, DataType::kF32);

  Blas::Set(spatial_dim, 1, sum_spatial_multiplier->mutable_data<float>(), 0,
            ws_->Ctx());

  Blas::BlasSgemv(0, batch * channel, spatial_dim, 1.f / spatial_dim,
                  bottom->data<float>(), 0,
                  sum_spatial_multiplier->data<float>(), 0, 0,
                  stats->mutable_data<float>(), 0, ws_->Ctx());
  Blas::BlasSgemm(0, 0, batch * channel, spatial_dim, 1, -1,
                  stats->data<float>(), 0,
                  sum_spatial_multiplier->data<float>(), 0, 1,
                  top->mutable_data<float>(), 0, ws_->Ctx());
  Blas::Pow(top->count(), top->data<float>(), 0, 2, temp->mutable_data<float>(),
            0, ws_->Ctx());
  Blas::BlasSgemv(0, batch * channel, spatial_dim, 1.f / spatial_dim,
                  temp->data<float>(), 0, sum_spatial_multiplier->data<float>(),
                  0, 0, stats->mutable_data<float>(), 0, ws_->Ctx());
  Blas::Add(stats->count(), stats->data<float>(), 0, eps_,
            stats->mutable_data<float>(), 0, ws_->Ctx());
  Blas::Pow(stats->count(), stats->data<float>(), 0, 0.5,
            stats->mutable_data<float>(), 0, ws_->Ctx());
  Blas::BlasSgemm(0, 0, batch * channel, spatial_dim, 1, 1,
                  stats->data<float>(), 0,
                  sum_spatial_multiplier->data<float>(), 0, 0,
                  temp->mutable_data<float>(), 0, ws_->Ctx());
  Blas::Div(top->count(), top->data<float>(), 0, temp->data<float>(), 0,
            top->mutable_data<float>(), 0, ws_->Ctx());

  if (bottoms_size() == 3) {
    const auto scale = bottoms(1);
    const auto bias = bottoms(2);
    CHECK_EQ(scale->count(), channel);
    CHECK_EQ(bias->count(), channel);
    Vision::Scale(top->data<float>(), top->count(), scale->data<float>(),
                  bias->data<float>(), channel, spatial_dim,
                  top->mutable_data<float>(), ws_->Ctx());
  }
#endif
}

REGISTER_OPERATOR(InstanceNorm, InstanceNormOp);

}  // namespace Shadow
