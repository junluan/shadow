#include "normalize_op.hpp"

namespace Shadow {

void NormalizeOp::Forward() {
  const auto bottom = bottoms(0);
  auto top = tops(0);

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
  ws_->GrowTempBuffer(temp_count * sizeof(float));

  std::shared_ptr<Blob> norm = nullptr, sum_channel_multiplier = nullptr;
  if (!across_spatial_) {
    norm = ws_->CreateTempBlob({1, 1, in_h, in_w}, DataType::kF32);
    sum_channel_multiplier =
        ws_->CreateTempBlob({1, in_c, 1, 1}, DataType::kF32);
    Blas::Set(in_c, 1, sum_channel_multiplier->mutable_data<float>(), 0,
              ws_->Ctx());
  }

  std::shared_ptr<Blob> sum_spatial_multiplier = nullptr;
  if (!channel_shared_ && bottoms_size() > 1) {
    sum_spatial_multiplier =
        ws_->CreateTempBlob({1, 1, in_h, in_w}, DataType::kF32);
    Blas::Set(spatial_dim, 1, sum_spatial_multiplier->mutable_data<float>(), 0,
              ws_->Ctx());
  }

  auto buffer = ws_->CreateTempBlob({1, in_c, in_h, in_w}, DataType::kF32);

  for (int b = 0; b < batch; ++b) {
    int data_offset = b * num;
    Blas::Square(num, bottom->data<float>(), data_offset,
                 buffer->mutable_data<float>(), 0, ws_->Ctx());
    if (across_spatial_) {
      float sum = 0;
      Blas::BlasSasum(num, buffer->data<float>(), 0, &sum, ws_->Ctx());
      float norm_value = std::sqrt(sum + EPS);
      Blas::Mul(num, bottom->data<float>(), data_offset, 1.f / norm_value,
                top->mutable_data<float>(), data_offset, ws_->Ctx());
    } else {
      Blas::Set(norm->count(), EPS, norm->mutable_data<float>(), 0, ws_->Ctx());
      Blas::BlasSgemv(1, in_c, spatial_dim, 1, buffer->data<float>(), 0,
                      sum_channel_multiplier->data<float>(), 0, 1,
                      norm->mutable_data<float>(), 0, ws_->Ctx());
      Blas::Pow(spatial_dim, norm->data<float>(), 0, 0.5f,
                norm->mutable_data<float>(), 0, ws_->Ctx());
      Blas::BlasSgemm(0, 0, in_c, spatial_dim, 1, 1,
                      sum_channel_multiplier->data<float>(), 0,
                      norm->data<float>(), 0, 0, buffer->mutable_data<float>(),
                      0, ws_->Ctx());
      Blas::Div(num, bottom->data<float>(), data_offset, buffer->data<float>(),
                0, top->mutable_data<float>(), data_offset, ws_->Ctx());
    }
    if (bottoms_size() > 1) {
      CHECK_EQ(bottoms_size(), 2);
      const auto scale = bottoms(1);
      if (channel_shared_) {
        CHECK_EQ(scale->count(), 1);
        float scale_data = 1;
        scale->get_data<float>(&scale_data, 1);
        Blas::BlasSscal(num, scale_data, top->mutable_data<float>(),
                        data_offset, ws_->Ctx());
      } else {
        CHECK_EQ(scale->count(), in_c);
        Blas::BlasSgemm(0, 0, in_c, spatial_dim, 1, 1, scale->data<float>(), 0,
                        sum_spatial_multiplier->data<float>(), 0, 0,
                        buffer->mutable_data<float>(), 0, ws_->Ctx());
        Blas::Mul(num, top->data<float>(), data_offset, buffer->data<float>(),
                  0, top->mutable_data<float>(), data_offset, ws_->Ctx());
      }
    }
  }
}

REGISTER_OPERATOR(Normalize, NormalizeOp);

}  // namespace Shadow
