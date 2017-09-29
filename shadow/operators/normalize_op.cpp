#include "normalize_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void NormalizeOp::Setup() {
  across_spatial_ = get_single_argument<bool>("across_spatial", true);
  channel_shared_ = get_single_argument<bool>("channel_shared", true);

  if (blobs_size() == 0) {
    add_blobs<float>(op_name_ + "_param_scale");
    auto *scale_blob = mutable_blobs<float>(0);
    if (channel_shared_) {
      scale_blob->reshape({1, 1});
    } else {
      scale_blob->reshape({1, bottoms<float>(0)->shape(1)});
    }
    Blas::Set(scale_blob->count(), 1, scale_blob->mutable_data(), 0);
    DLOG(WARNING) << "Scale param is initialized with the default values 1";
  }

  CHECK_EQ(blobs_size(), 1);
  if (channel_shared_) {
    CHECK_EQ(blobs<float>(0)->count(), 1);
  } else {
    CHECK_EQ(blobs<float>(0)->count(), bottoms<float>(0)->shape(1));
  }
}

void NormalizeOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);

  top->reshape(bottom->shape());

  spatial_dim_ = in_h * in_w;

  if (!across_spatial_) {
    norm_ = op_ws_->CreateBlob<float>(op_name_ + "_norm");
    norm_->reshape({1, 1, in_h, in_w});
    sum_channel_multiplier_ =
        op_ws_->CreateBlob<float>(op_name_ + "_sum_channel_multiplier");
    sum_channel_multiplier_->reshape({1, in_c, 1, 1});
    Blas::Set(in_c, 1, sum_channel_multiplier_->mutable_data(), 0);
  }

  if (!channel_shared_) {
    sum_spatial_multiplier_ =
        op_ws_->CreateBlob<float>(op_name_ + "_sum_spatial_multiplier");
    sum_spatial_multiplier_->reshape({1, 1, in_h, in_w});
    Blas::Set(spatial_dim_, 1, sum_spatial_multiplier_->mutable_data(), 0);
  }

  buffer_ = op_ws_->CreateBlob<float>(op_name_ + "_buffer");
  buffer_->reshape({1, in_c, in_h, in_w});

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void NormalizeOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  int batch = bottom->shape(0), channels = bottom->shape(1);
  int num = bottom->num();
  for (int b = 0; b < batch; ++b) {
    int data_offset = b * num;
    Blas::Square(num, bottom->data(), data_offset, buffer_->mutable_data(), 0);
    if (across_spatial_) {
      float sum = 0;
      Blas::BlasSasum(num, buffer_->data(), 0, &sum);
      float norm = std::sqrt(sum + EPS);
      Blas::Mul(num, bottom->data(), data_offset, 1.f / norm,
                top->mutable_data(), data_offset);
    } else {
      Blas::Set(norm_->count(), EPS, norm_->mutable_data(), 0);
      Blas::BlasSgemv(1, channels, spatial_dim_, 1, buffer_->data(), 0,
                      sum_channel_multiplier_->data(), 0, 1,
                      norm_->mutable_data(), 0);
      Blas::Pow(spatial_dim_, norm_->data(), 0, 0.5f, norm_->mutable_data(), 0);
      Blas::BlasSgemm(0, 0, channels, spatial_dim_, 1, 1,
                      sum_channel_multiplier_->data(), 0, norm_->data(), 0, 0,
                      buffer_->mutable_data(), 0);
      Blas::Div(num, bottom->data(), data_offset, buffer_->data(), 0,
                top->mutable_data(), data_offset);
    }
    if (channel_shared_) {
      blobs<float>(0)->read_data(&scale_, 1);
      Blas::BlasSscal(num, scale_, top->mutable_data(), data_offset);
    } else {
      Blas::BlasSgemm(
          0, 0, channels, spatial_dim_, 1, 1, blobs<float>(0)->data(), 0,
          sum_spatial_multiplier_->data(), 0, 0, buffer_->mutable_data(), 0);
      Blas::Mul(num, top->data(), data_offset, buffer_->data(), 0,
                top->mutable_data(), data_offset);
    }
  }
}

void NormalizeOp::Release() {
  // DLOG(INFO) << "Free NormalizeOp!";
}

REGISTER_OPERATOR(Normalize, NormalizeOp);

}  // namespace Shadow
