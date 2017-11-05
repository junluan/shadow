#ifndef SHADOW_OPERATORS_SCALE_OP_HPP
#define SHADOW_OPERATORS_SCALE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ScaleOp : public Operator {
 public:
  explicit ScaleOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);
    axis_ = bottoms<float>(0)->canonical_index(axis_);
    num_axis_ = get_single_argument<int>("num_axis", 1);
    CHECK_GE(num_axis_, -1);
    bias_term_ = get_single_argument<bool>("bias_term", false);

    if (bottoms_size() == 1) {
      CHECK_GE(blobs_size(), 1);
      scale_ = const_cast<BlobF *>(blobs<float>(0));
    } else {
      scale_ = const_cast<BlobF *>(bottoms<float>(1));
    }

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
    bias_ = const_cast<BlobF *>(blobs<float>(bias_param_id_));
  }

  void Reshape() override;
  void Forward() override;

 private:
  bool bias_term_;
  int axis_, num_axis_, scale_dim_, inner_dim_, bias_param_id_;

  BlobF *scale_ = nullptr, *bias_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SCALE_OP_HPP
