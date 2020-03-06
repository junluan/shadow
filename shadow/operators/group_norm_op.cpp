#include "group_norm_op.hpp"

#include "scale_op.hpp"

namespace Shadow {

void GroupNormOp::Forward() {
  CHECK(bottoms_size() == 1 || bottoms_size() == 3);

  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  top->reshape(bottom->shape());

  int batch = bottom->shape(0), channel = bottom->shape(1),
      spatial_dim = bottom->count(2);

  CHECK_EQ(channel % group_, 0);

  int temp_count =
      2 * batch * group_ + bottom->count() + batch * channel + spatial_dim;
  op_ws_->GrowTempBuffer(temp_count, sizeof(float));

  auto *mean =
      op_ws_->CreateTempBlob<float>({batch, group_}, op_name_ + "/mean");
  auto *variance =
      op_ws_->CreateTempBlob<float>({batch, group_}, op_name_ + "/variance");
  auto *temp =
      op_ws_->CreateTempBlob<float>(bottom->shape(), op_name_ + "/temp");
  auto *batch_by_channel = op_ws_->CreateTempBlob<float>(
      {batch, channel}, op_name_ + "/batch_by_channel");
  auto *sum_spatial_multiplier = op_ws_->CreateTempBlob<float>(
      {1, 1, spatial_dim}, op_name_ + "/sum_spatial_multiplier");

  Blas::Set(spatial_dim, 1, sum_spatial_multiplier->mutable_data(), 0,
            op_ws_->Ctx());

  Blas::BlasSgemv(0, batch * channel, spatial_dim, 1.f / spatial_dim,
                  bottom->data(), 0, sum_spatial_multiplier->data(), 0, 0,
                  batch_by_channel->mutable_data(), 0, op_ws_->Ctx());
  Vision::ComputeGroup(batch_by_channel->data(), batch, channel, group_,
                       mean->mutable_data(), op_ws_->Ctx());

  Vision::SubtractMean(bottom->data(), mean->data(), batch, channel,
                       spatial_dim, group_, top->mutable_data(), op_ws_->Ctx());

  Blas::Pow(top->count(), top->data(), 0, 2, temp->mutable_data(), 0,
            op_ws_->Ctx());

  Blas::BlasSgemv(0, batch * channel, spatial_dim, 1.f / spatial_dim,
                  temp->data(), 0, sum_spatial_multiplier->data(), 0, 0,
                  batch_by_channel->mutable_data(), 0, op_ws_->Ctx());
  Vision::ComputeGroup(batch_by_channel->data(), batch, channel, group_,
                       variance->mutable_data(), op_ws_->Ctx());

  Vision::DivideVariance(top->data(), variance->data(), batch, channel,
                         spatial_dim, group_, eps_, top->mutable_data(),
                         op_ws_->Ctx());

  if (bottoms_size() == 3) {
    const auto *scale = bottoms<float>(1);
    const auto *bias = bottoms<float>(2);
    CHECK_EQ(scale->count(), channel);
    CHECK_EQ(bias->count(), channel);
    Vision::Scale(top->data(), top->count(), scale->data(), bias->data(),
                  channel, spatial_dim, top->mutable_data(), op_ws_->Ctx());
  }
}

REGISTER_OPERATOR(GroupNorm, GroupNormOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void ComputeGroup(const T *in_data, int batch, int channel, int group,
                  T *out_data, Context *context) {
  int num_val = channel / group;
  for (int b = 0; b < batch; ++b) {
    for (int g = 0; g < group; ++g) {
      T sum = T(0);
      for (int n = 0; n < num_val; ++n, in_data++) {
        sum += *in_data;
      }
      *out_data++ = sum / num_val;
    }
  }
}

template <typename T>
void SubtractMean(const T *in_data, const T *mean_data, int batch, int channel,
                  int spatial_dim, int group, T *out_data, Context *context) {
  int num_val = channel / group;
  for (int b = 0; b < batch; ++b, mean_data += group) {
    for (int c = 0; c < channel; ++c) {
      T mean = mean_data[c / num_val];
      for (int n = 0; n < spatial_dim; ++n) {
        *out_data++ = *in_data++ - mean;
      }
    }
  }
}

template <typename T>
void DivideVariance(const T *in_data, const T *variance_data, int batch,
                    int channel, int spatial_dim, int group, float eps,
                    T *out_data, Context *context) {
  int num_val = channel / group;
  for (int b = 0; b < batch; ++b, variance_data += group) {
    for (int c = 0; c < channel; ++c) {
      T variance = variance_data[c / num_val];
      for (int n = 0; n < spatial_dim; ++n) {
        *out_data++ = *in_data++ / std::sqrt(variance + eps);
      }
    }
  }
}

template void ComputeGroup(const float *, int, int, int, float *, Context *);
template void SubtractMean(const float *, const float *, int, int, int, int,
                           float *, Context *);
template void DivideVariance(const float *, const float *, int, int, int, int,
                             float, float *, Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
