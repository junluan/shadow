#include "normalize_op.hpp"

namespace Shadow {

void NormalizeOp::Forward() {
  CHECK_EQ(bottoms_size(), 2);

  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0), in_c = bottom->shape(1),
      in_h = bottom->shape(2), in_w = bottom->shape(3);
  int num = bottom->num(), spatial_dim = in_h * in_w;

  top->reshape(bottom->shape());

  int temp_count = in_c * spatial_dim;
  if (!across_spatial_) {
    temp_count += spatial_dim;
    temp_count += in_c;
  }
  if (!channel_shared_ && bottoms_size() > 1) {
    temp_count += spatial_dim;
  }
  op_ws_->GrowTempBuffer(temp_count, sizeof(float));

  if (!across_spatial_) {
    norm_ =
        op_ws_->CreateTempBlob<float>({1, 1, in_h, in_w}, op_name_ + "_norm");
    sum_channel_multiplier_ = op_ws_->CreateTempBlob<float>(
        {1, in_c, 1, 1}, op_name_ + "_sum_channel_multiplier");
    Blas::Set(in_c, 1, sum_channel_multiplier_->mutable_data(), 0);
  }

  if (!channel_shared_ && bottoms_size() > 1) {
    sum_spatial_multiplier_ = op_ws_->CreateTempBlob<float>(
        {1, 1, in_h, in_w}, op_name_ + "_sum_spatial_multiplier");
    Blas::Set(spatial_dim, 1, sum_spatial_multiplier_->mutable_data(), 0);
  }

  buffer_ = op_ws_->CreateTempBlob<float>({1, in_c, in_h, in_w},
                                          op_name_ + "_buffer");

  for (int b = 0; b < batch; ++b) {
    int data_offset = b * num;
    Blas::Square(num, bottom->data(), data_offset, buffer_->mutable_data(), 0);
    if (across_spatial_) {
      float sum = 0;
      Blas::BlasSasum(num, buffer_->data(), 0, &sum,
                      op_ws_->Ctx()->blas_handle());
      float norm = std::sqrt(sum + EPS);
      Blas::Mul(num, bottom->data(), data_offset, 1.f / norm,
                top->mutable_data(), data_offset);
    } else {
      Blas::Set(norm_->count(), EPS, norm_->mutable_data(), 0);
      Blas::BlasSgemv(1, in_c, spatial_dim, 1, buffer_->data(), 0,
                      sum_channel_multiplier_->data(), 0, 1,
                      norm_->mutable_data(), 0, op_ws_->Ctx()->blas_handle());
      Blas::Pow(spatial_dim, norm_->data(), 0, 0.5f, norm_->mutable_data(), 0);
      Blas::BlasSgemm(0, 0, in_c, spatial_dim, 1, 1,
                      sum_channel_multiplier_->data(), 0, norm_->data(), 0, 0,
                      buffer_->mutable_data(), 0, op_ws_->Ctx()->blas_handle());
      Blas::Div(num, bottom->data(), data_offset, buffer_->data(), 0,
                top->mutable_data(), data_offset);
    }
    if (bottoms_size() > 1) {
      CHECK_EQ(bottoms_size(), 2);
      const auto *scale = bottoms<float>(1);
      if (channel_shared_) {
        CHECK_EQ(scale->count(), 1);
        float scale_data = 1;
        scale->read_data(&scale_data, 1);
        Blas::BlasSscal(num, scale_data, top->mutable_data(), data_offset,
                        op_ws_->Ctx()->blas_handle());
      } else {
        CHECK_EQ(scale->count(), in_c);
        Blas::BlasSgemm(0, 0, in_c, spatial_dim, 1, 1, scale->data(), 0,
                        sum_spatial_multiplier_->data(), 0, 0,
                        buffer_->mutable_data(), 0,
                        op_ws_->Ctx()->blas_handle());
        Blas::Mul(num, top->data(), data_offset, buffer_->data(), 0,
                  top->mutable_data(), data_offset);
      }
    }
  }
}

REGISTER_OPERATOR(Normalize, NormalizeOp);

}  // namespace Shadow
