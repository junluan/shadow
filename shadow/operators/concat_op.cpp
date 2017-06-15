#include "concat_op.hpp"
#include "core/image.hpp"

namespace Shadow {

void ConcatOp::Setup(VecBlobF *blobs) {
  Operator::Setup(blobs);

  const auto &concat_param = op_param_.concat_param();

  concat_axis_ = concat_param.axis();
  CHECK_GE(concat_axis_, 0);
  CHECK_LT(concat_axis_, bottoms_[0]->num_axes());
  num_concats_ = bottoms_[0]->count(0, concat_axis_);
  concat_input_size_ = bottoms_[0]->count(concat_axis_ + 1);
}

void ConcatOp::Reshape() {
  int num_axes = bottoms_[0]->num_axes();
  VecInt top_shape = bottoms_[0]->shape();
  for (int i = 1; i < bottoms_.size(); ++i) {
    CHECK_EQ(num_axes, bottoms_[i]->num_axes())
        << "Bottoms must have the same axes!";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) continue;
      CHECK_EQ(top_shape[j], bottoms_[i]->shape(j))
          << "Bottoms must have the same shape, except at concat_axis!";
    }
    top_shape[concat_axis_] += bottoms_[i]->shape(concat_axis_);
  }
  tops_[0]->set_shape(top_shape);
  if (bottoms_.size() > 1) {
    tops_[0]->reshape(top_shape);
  }

  VecString str;
  for (const auto &bottom : bottoms_) {
    str.push_back(Util::format_vector(bottom->shape(), ",", "(", ")"));
  }
  DLOG(INFO) << op_name_ << ": " << Util::format_vector(str, " + ") << " -> "
             << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void ConcatOp::Forward() {
  if (bottoms_.size() == 1) {
    tops_[0]->share_data(bottoms_[0]->data());
    return;
  }
  int offset_concat_axis = 0;
  int top_concat_axis = tops_[0]->shape(concat_axis_);
  for (const auto &bottom : bottoms_) {
    int bottom_concat_axis = bottom->shape(concat_axis_);
    Image::Concat(bottom->data(), bottom->count(), num_concats_,
                  concat_input_size_, top_concat_axis, bottom_concat_axis,
                  offset_concat_axis, tops_[0]->mutable_data());
    offset_concat_axis += bottom_concat_axis;
  }
}

void ConcatOp::Release() {
  // DLOG(INFO) << "Free ConcatOp!";
}

}  // namespace Shadow
