#include "bias_op.hpp"
#include "core/blas.hpp"
#include "core/vision.hpp"

namespace Shadow {

void BiasOp::Setup() {
  axis_ = get_single_argument<int>("axis", 1);
  axis_ = bottoms<float>(0)->canonical_index(axis_);
  num_axis_ = get_single_argument<int>("num_axis", 1);
  CHECK_GE(num_axis_, -1);

  if (bottoms_size() == 1 && blobs_size() == 0) {
    int end_axis;
    if (num_axis_ == -1) {
      end_axis = bottoms<float>(0)->num_axes();
    } else {
      end_axis = axis_ + num_axis_;
      CHECK_GE(bottoms<float>(0)->num_axes(), end_axis);
    }
    VecInt bias_shape;
    for (int i = axis_; i < end_axis; ++i) {
      bias_shape.push_back(bottoms<float>(0)->shape(i));
    }
    add_blobs<float>(op_name_ + "_param_bias");
    auto *bias_blob = mutable_blobs<float>(0);
    bias_blob->reshape(bias_shape);
    Blas::Set(bias_blob->count(), 0, bias_blob->mutable_data(), 0);
    DLOG(WARNING) << "Bias param is initialized with the default value 0";
  }
  bias_ =
      bottoms_size() > 1 ? mutable_bottoms<float>(1) : mutable_blobs<float>(0);
}

void BiasOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  int start_axis = bias_->num_axes() == 0 ? 0 : axis_;
  CHECK_GE(bottom->num_axes(), start_axis + bias_->num_axes());
  for (int i = 0; i < bias_->num_axes(); ++i) {
    CHECK_EQ(bottom->shape(start_axis + i), bias_->shape(i));
  }
  bias_dim_ = bias_->count();
  inner_dim_ = bottom->count(start_axis + bias_->num_axes());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void BiasOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::Bias(bottom->data(), bottom->count(), bias_->data(), bias_dim_,
               inner_dim_, top->mutable_data());
}

void BiasOp::Release() {
  // DLOG(INFO) << "Free BiasOp!";
}

REGISTER_OPERATOR(Bias, BiasOp);

}  // namespace Shadow
