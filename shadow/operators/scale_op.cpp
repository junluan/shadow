#include "scale_op.hpp"
#include "core/blas.hpp"
#include "core/image.hpp"

namespace Shadow {

void ScaleOp::Setup(VecBlobF *blobs) {
  Operator::Setup(blobs);

  const auto &scale_param = op_param_.scale_param();

  axis_ = bottoms_[0]->canonical_index(scale_param.axis());
  num_axis_ = scale_param.num_axes();
  CHECK_GE(num_axis_, -1);
  bias_term_ = scale_param.bias_term();

  if (bottoms_.size() == 1 && blobs_.size() == 0) {
    int end_axis;
    if (num_axis_ == -1) {
      end_axis = bottoms_[0]->num_axes();
    } else {
      end_axis = axis_ + num_axis_;
      CHECK_GE(bottoms_[0]->num_axes(), end_axis);
    }
    VecInt scale_shape;
    for (int i = axis_; i < end_axis; ++i) {
      scale_shape.push_back(bottoms_[0]->shape(i));
    }
    blobs_.push_back(new BlobF(scale_shape));
    Blas::Set(blobs_[0]->count(), 1, blobs_[0]->mutable_data(), 0);
    DLOG(WARNING) << "Scale param is initialized with the default value 1";
  }
  scale_ = bottoms_.size() > 1 ? bottoms_[1] : blobs_[0];

  if (bias_term_ && (bottoms_.size() + blobs_.size() > 2)) {
    bias_param_id_ = blobs_.size() - 1;
  } else {
    bias_param_id_ = blobs_.size();
    blobs_.resize(bias_param_id_ + 1);
    blobs_[bias_param_id_] = new BlobF(scale_->shape());
    Blas::Set(blobs_[bias_param_id_]->count(), 0,
              blobs_[bias_param_id_]->mutable_data(), 0);
    DLOG(WARNING) << "Bias param is initialized with the default value 0";
  }
  bias_ = blobs_[bias_param_id_];
}

void ScaleOp::Reshape() {
  if (bottoms_[0] != tops_[0]) {
    tops_[0]->reshape(bottoms_[0]->shape());
  }

  int start_axis = scale_->num_axes() == 0 ? 0 : axis_;
  CHECK_GE(bottoms_[0]->num_axes(), start_axis + scale_->num_axes());
  for (int i = 0; i < scale_->num_axes(); ++i) {
    CHECK_EQ(bottoms_[0]->shape(start_axis + i), scale_->shape(i));
  }
  scale_dim_ = scale_->count();
  inner_dim_ = bottoms_[0]->count(start_axis + scale_->num_axes());

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void ScaleOp::Forward() {
  Image::Scale(bottoms_[0]->data(), bottoms_[0]->count(), scale_->data(),
               bias_->data(), scale_dim_, inner_dim_, tops_[0]->mutable_data());
}

void ScaleOp::Release() {
  // DLOG(INFO) << "Free ScaleOp!";
}

REGISTER_OP_CLASS(Scale);

}  // namespace Shadow
