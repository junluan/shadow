#include "concat_op.hpp"
#include "core/image.hpp"

namespace Shadow {

void ConcatOp::Setup() {
  concat_axis_ = get_single_argument<int>("axis", 1);
  CHECK_GE(concat_axis_, 0);
  CHECK_LT(concat_axis_, bottoms<float>(0)->num_axes());
  num_concats_ = bottoms<float>(0)->count(0, concat_axis_);
  concat_input_size_ = bottoms<float>(0)->count(concat_axis_ + 1);
}

void ConcatOp::Reshape() {
  auto *top = mutable_tops<float>(0);

  int num_axes = bottoms<float>(0)->num_axes();
  auto top_shape = bottoms<float>(0)->shape();
  for (int i = 1; i < bottoms_size(); ++i) {
    CHECK_EQ(num_axes, bottoms<float>(i)->num_axes())
        << "Bottoms must have the same axes!";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) continue;
      CHECK_EQ(top_shape[j], bottoms<float>(i)->shape(j))
          << "Bottoms must have the same shape, except at concat_axis!";
    }
    top_shape[concat_axis_] += bottoms<float>(i)->shape(concat_axis_);
  }
  top->set_shape(top_shape);
  if (bottoms_size() > 1) {
    top->reshape(top_shape);
  }

  VecString str;
  for (int i = 0; i < bottoms_size(); ++i) {
    str.push_back(
        Util::format_vector(bottoms<float>(i)->shape(), ",", "(", ")"));
  }
  DLOG(INFO) << op_name_ << ": " << Util::format_vector(str, " + ") << " -> "
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void ConcatOp::Forward() {
  auto *top = mutable_tops<float>(0);

  if (bottoms_size() == 1) {
    top->share_data(*bottoms<float>(0));
    return;
  }
  int offset_concat_axis = 0;
  int top_concat_axis = top->shape(concat_axis_);
  for (int i = 0; i < bottoms_size(); ++i) {
    const auto *bottom = bottoms<float>(i);
    int bottom_concat_axis = bottom->shape(concat_axis_);
    Image::Concat(bottom->data(), bottom->count(), num_concats_,
                  concat_input_size_, top_concat_axis, bottom_concat_axis,
                  offset_concat_axis, top->mutable_data());
    offset_concat_axis += bottom_concat_axis;
  }
}

void ConcatOp::Release() {
  // DLOG(INFO) << "Free ConcatOp!";
}

REGISTER_OPERATOR(Concat, ConcatOp);

}  // namespace Shadow
