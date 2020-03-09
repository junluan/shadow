#include "scale_op.hpp"

namespace Shadow {

void ScaleOp::Forward() {
  const auto bottom = bottoms(0);
  auto top = tops(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  std::shared_ptr<Blob> scale = nullptr, bias = nullptr;
  if (scale_value_.empty() && bias_value_.empty()) {
    CHECK_GE(bottoms_size(), 2);
    if (has_scale_ && has_bias_) {
      CHECK_EQ(bottoms_size(), 3);
      scale = bottoms(1), bias = bottoms(2);
    } else if (has_scale_) {
      scale = bottoms(1);
      ws_->GrowTempBuffer(scale->raw_size());
      bias = ws_->CreateTempBlob(scale->shape(), DataType::kF32);
      Blas::Set(bias->count(), 0, bias->mutable_data<float>(), 0, ws_->Ctx());
    } else {
      bias = bottoms(1);
      ws_->GrowTempBuffer(bias->raw_size());
      scale = ws_->CreateTempBlob(bias->shape(), DataType::kF32);
      Blas::Set(scale->count(), 1, scale->mutable_data<float>(), 0, ws_->Ctx());
    }
  } else {
    int dim = bottom->shape(axis_);
    if (scale_value_.size() > 1) {
      CHECK_EQ(scale_value_.size(), dim);
    } else if (scale_value_.size() == 1) {
      scale_value_ = VecFloat(dim, scale_value_[0]);
    } else {
      scale_value_ = VecFloat(dim, 1);
    }
    if (bias_value_.size() > 1) {
      CHECK_EQ(bias_value_.size(), dim);
    } else if (bias_value_.size() == 1) {
      bias_value_ = VecFloat(dim, bias_value_[0]);
    } else {
      bias_value_ = VecFloat(dim, 0);
    }
    ws_->GrowTempBuffer(2 * dim * sizeof(float));
    scale = ws_->CreateTempBlob({dim}, DataType::kF32);
    bias = ws_->CreateTempBlob({dim}, DataType::kF32);
    scale->set_data<float>(scale_value_.data(), dim);
    bias->set_data<float>(bias_value_.data(), dim);
  }

  VecInt shape;
  for (int d = scale->num_axes() - 1; d >= 0; --d) {
    int dim = scale->shape(d);
    if (dim != 1 || !shape.empty() || d == 0) {
      shape.insert(shape.begin(), dim);
    }
  }
  scale->set_shape(shape);
  bias->set_shape(shape);

  CHECK_GE(bottom->num_axes(), axis_ + scale->num_axes());
  for (int d = 0; d < scale->num_axes(); ++d) {
    CHECK_EQ(bottom->shape(axis_ + d), scale->shape(d));
  }
  scale_dim_ = scale->count();
  inner_dim_ = bottom->count(axis_ + scale->num_axes());

  Vision::Scale(bottom->data<float>(), bottom->count(), scale->data<float>(),
                bias->data<float>(), scale_dim_, inner_dim_,
                top->mutable_data<float>(), ws_->Ctx());
}

REGISTER_OPERATOR(Scale, ScaleOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data, Context *context) {
  for (int i = 0; i < count; ++i) {
    int index = (i / inner_dim) % scale_dim;
    out_data[i] = in_data[i] * scale_data[index] + bias_data[index];
  }
}

template void Scale(const float *, int, const float *, const float *, int, int,
                    float *, Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
