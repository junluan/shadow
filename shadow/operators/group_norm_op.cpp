#include "group_norm_op.hpp"

#include "scale_op.hpp"

namespace Shadow {

void GroupNormOp::Forward() {
  CHECK(bottoms_size() == 1 || bottoms_size() == 3);

  const auto bottom = bottoms(0);
  auto top = tops(0);

  top->reshape(bottom->shape());

  int batch = bottom->shape(0), channel = bottom->shape(1),
      spatial_dim = bottom->count(2);

  CHECK_EQ(channel % group_, 0);

  int temp_count =
      2 * batch * group_ + bottom->count() + batch * channel + spatial_dim;
  ws_->GrowTempBuffer(temp_count * sizeof(float));

  auto mean = ws_->CreateTempBlob({batch, group_}, DataType::kF32);
  auto variance = ws_->CreateTempBlob({batch, group_}, DataType::kF32);
  auto temp = ws_->CreateTempBlob(bottom->shape(), DataType::kF32);
  auto batch_by_channel = ws_->CreateTempBlob({batch, channel}, DataType::kF32);
  auto sum_spatial_multiplier =
      ws_->CreateTempBlob({1, 1, spatial_dim}, DataType::kF32);

  Blas::Set(spatial_dim, 1, sum_spatial_multiplier->mutable_data<float>(), 0,
            ws_->Ctx());

  Blas::BlasSgemv(0, batch * channel, spatial_dim, 1.f / spatial_dim,
                  bottom->data<float>(), 0,
                  sum_spatial_multiplier->data<float>(), 0, 0,
                  batch_by_channel->mutable_data<float>(), 0, ws_->Ctx());
  Vision::ComputeGroup(batch_by_channel->data<float>(), batch, channel, group_,
                       mean->mutable_data<float>(), ws_->Ctx());

  Vision::SubtractMean(bottom->data<float>(), mean->data<float>(), batch,
                       channel, spatial_dim, group_, top->mutable_data<float>(),
                       ws_->Ctx());

  Blas::Pow(top->count(), top->data<float>(), 0, 2, temp->mutable_data<float>(),
            0, ws_->Ctx());

  Blas::BlasSgemv(0, batch * channel, spatial_dim, 1.f / spatial_dim,
                  temp->data<float>(), 0, sum_spatial_multiplier->data<float>(),
                  0, 0, batch_by_channel->mutable_data<float>(), 0, ws_->Ctx());
  Vision::ComputeGroup(batch_by_channel->data<float>(), batch, channel, group_,
                       variance->mutable_data<float>(), ws_->Ctx());

  Vision::DivideVariance(top->data<float>(), variance->data<float>(), batch,
                         channel, spatial_dim, group_, eps_,
                         top->mutable_data<float>(), ws_->Ctx());

  if (bottoms_size() == 3) {
    const auto scale = bottoms(1);
    const auto bias = bottoms(2);
    CHECK_EQ(scale->count(), channel);
    CHECK_EQ(bias->count(), channel);
    Vision::Scale(top->data<float>(), top->count(), scale->data<float>(),
                  bias->data<float>(), channel, spatial_dim,
                  top->mutable_data<float>(), ws_->Ctx());
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
