#include "instance_norm_op.hpp"

#include "scale_op.hpp"

namespace Shadow {

void InstanceNormOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0,
                    op_ws_->Ctx()->blas_handle());
  }

  int batch = bottom->shape(0), channel = bottom->shape(1),
      spatial_dim = bottom->count(2);

  int temp_count = batch * channel + bottom->count() + spatial_dim;
  op_ws_->GrowTempBuffer(temp_count, sizeof(float));

  auto *stats =
      op_ws_->CreateTempBlob<float>({batch, channel}, op_name_ + "/stats");
  auto *temp =
      op_ws_->CreateTempBlob<float>(bottom->shape(), op_name_ + "/temp");
  auto *sum_spatial_multiplier = op_ws_->CreateTempBlob<float>(
      {1, 1, spatial_dim}, op_name_ + "/sum_spatial_multiplier");

  Blas::Set(spatial_dim, 1, sum_spatial_multiplier->mutable_data(), 0);

  Blas::BlasSgemv(0, batch * channel, spatial_dim, 1.f / spatial_dim,
                  bottom->data(), 0, sum_spatial_multiplier->data(), 0, 0,
                  stats->mutable_data(), 0, op_ws_->Ctx()->blas_handle());
  Blas::BlasSgemm(0, 0, batch * channel, spatial_dim, 1, -1, stats->data(), 0,
                  sum_spatial_multiplier->data(), 0, 1, top->mutable_data(), 0,
                  op_ws_->Ctx()->blas_handle());
  Blas::Pow(top->count(), top->data(), 0, 2, temp->mutable_data(), 0);
  Blas::BlasSgemv(0, batch * channel, spatial_dim, 1.f / spatial_dim,
                  temp->data(), 0, sum_spatial_multiplier->data(), 0, 0,
                  stats->mutable_data(), 0, op_ws_->Ctx()->blas_handle());
  Blas::Add(stats->count(), stats->data(), 0, eps_, stats->mutable_data(), 0);
  Blas::Pow(stats->count(), stats->data(), 0, 0.5, stats->mutable_data(), 0);
  Blas::BlasSgemm(0, 0, batch * channel, spatial_dim, 1, 1, stats->data(), 0,
                  sum_spatial_multiplier->data(), 0, 0, temp->mutable_data(), 0,
                  op_ws_->Ctx()->blas_handle());
  Blas::Div(top->count(), top->data(), 0, temp->data(), 0, top->mutable_data(),
            0);

  if (bottoms_size() > 1) {
    CHECK_EQ(bottoms_size(), 3);
    const auto *scale = bottoms<float>(1);
    const auto *bias = bottoms<float>(2);
    CHECK_EQ(scale->count(), channel);
    CHECK_EQ(bias->count(), channel);
    Vision::Scale(top->data(), top->count(), scale->data(), bias->data(),
                  channel, spatial_dim, top->mutable_data());
  }
}

REGISTER_OPERATOR(InstanceNorm, InstanceNormOp);

}  // namespace Shadow
