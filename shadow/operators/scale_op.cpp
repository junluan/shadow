#include "scale_op.hpp"
#include "core/blas.hpp"
#include "core/image.hpp"

namespace Shadow {

void ScaleOp::Setup() {
  axis_ = get_single_argument<int>("axis", 1);
  axis_ = bottoms<float>(0)->canonical_index(axis_);
  num_axis_ = get_single_argument<int>("num_axis", 1);
  CHECK_GE(num_axis_, -1);
  bias_term_ = get_single_argument<bool>("bias_term", false);

  if (bottoms_size() == 1 && blobs_size() == 0) {
    int end_axis;
    if (num_axis_ == -1) {
      end_axis = bottoms<float>(0)->num_axes();
    } else {
      end_axis = axis_ + num_axis_;
      CHECK_GE(bottoms<float>(0)->num_axes(), end_axis);
    }
    VecInt scale_shape;
    for (int i = axis_; i < end_axis; ++i) {
      scale_shape.push_back(bottoms<float>(0)->shape(i));
    }
    add_blobs<float>(op_name_ + "_param_scale");
    auto *scale_blob = mutable_blobs<float>(0);
    scale_blob->reshape(scale_shape);
    Blas::Set(scale_blob->count(), 1, scale_blob->mutable_data(), 0);
    DLOG(WARNING) << "Scale param is initialized with the default value 1";
  }
  scale_ =
      bottoms_size() > 1 ? mutable_bottoms<float>(1) : mutable_blobs<float>(0);

  if (bias_term_ && (bottoms_size() + blobs_size() > 2)) {
    bias_param_id_ = blobs_size() - 1;
  } else {
    bias_param_id_ = blobs_size();
    add_blobs<float>(op_name_ + "_param_bias");
    auto *bias_blob = mutable_blobs<float>(bias_param_id_);
    bias_blob->reshape(scale_->shape());
    Blas::Set(bias_blob->count(), 0, bias_blob->mutable_data(), 0);
    DLOG(WARNING) << "Bias param is initialized with the default value 0";
  }
  bias_ = mutable_blobs<float>(bias_param_id_);
}

void ScaleOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  int start_axis = scale_->num_axes() == 0 ? 0 : axis_;
  CHECK_GE(bottom->num_axes(), start_axis + scale_->num_axes());
  for (int i = 0; i < scale_->num_axes(); ++i) {
    CHECK_EQ(bottom->shape(start_axis + i), scale_->shape(i));
  }
  scale_dim_ = scale_->count();
  inner_dim_ = bottom->count(start_axis + scale_->num_axes());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void ScaleOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Image::Scale(bottom->data(), bottom->count(), scale_->data(), bias_->data(),
               scale_dim_, inner_dim_, top->mutable_data());
}

void ScaleOp::Release() {
  // DLOG(INFO) << "Free ScaleOp!";
}

REGISTER_OPERATOR(Scale, ScaleOp);

}  // namespace Shadow
