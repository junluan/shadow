#include "scale_op.hpp"

namespace Shadow {

void ScaleOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  BlobF *scale = nullptr, *bias = nullptr;
  if (scale_value_.empty() && bias_value_.empty()) {
    CHECK_GE(bottoms_size(), 2);
    if (has_scale_ && has_bias_) {
      CHECK_EQ(bottoms_size(), 3);
      scale = const_cast<BlobF *>(bottoms<float>(1));
      bias = const_cast<BlobF *>(bottoms<float>(2));
    } else if (has_scale_) {
      scale = const_cast<BlobF *>(bottoms<float>(1));
      op_ws_->GrowTempBuffer(scale->count(), sizeof(float));
      bias = op_ws_->CreateTempBlob<float>(scale->shape(),
                                           op_name_ + "/bias_value");
      Blas::Set(bias->count(), 0, bias->mutable_data(), 0);
    } else {
      bias = const_cast<BlobF *>(bottoms<float>(1));
      op_ws_->GrowTempBuffer(bias->count(), sizeof(float));
      scale = op_ws_->CreateTempBlob<float>(bias->shape(),
                                            op_name_ + "/scale_value");
      Blas::Set(scale->count(), 1, scale->mutable_data(), 0);
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
    op_ws_->GrowTempBuffer(2 * dim, sizeof(float));
    scale = op_ws_->CreateTempBlob<float>({dim}, op_name_ + "/scale_value");
    bias = op_ws_->CreateTempBlob<float>({dim}, op_name_ + "/bias_value");
    scale->set_data(scale_value_.data(), dim);
    bias->set_data(bias_value_.data(), dim);
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

  Vision::Scale(bottom->data(), bottom->count(), scale->data(), bias->data(),
                scale_dim_, inner_dim_, top->mutable_data());
}

REGISTER_OPERATOR(Scale, ScaleOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data) {
  for (int i = 0; i < count; ++i) {
    int index = (i / inner_dim) % scale_dim;
    out_data[i] = in_data[i] * scale_data[index] + bias_data[index];
  }
}

template void Scale(const float *in_data, int count, const float *scale_data,
                    const float *bias_data, int scale_dim, int inner_dim,
                    float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
