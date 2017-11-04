#include "depthwise_conv_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

void DepthwiseConvOp::Setup() {
  num_output_ = get_single_argument<int>("num_output", 0);
  CHECK(has_argument("kernel_size"));
  kernel_size_ = get_single_argument<int>("kernel_size", 0);
  stride_ = get_single_argument<int>("stride", 1);
  pad_ = get_single_argument<int>("pad", 0);
  dilation_ = get_single_argument<int>("dilation", 1);
  CHECK_EQ(dilation_, 1);
  group_ = get_single_argument<int>("group", 1);
  CHECK_EQ(bottoms<float>(0)->shape(1), group_);
  CHECK_EQ(num_output_, group_);
  bias_term_ = get_single_argument<bool>("bias_term", true);
  activate_type_ = get_single_argument<int>("type", -1);
  CHECK((activate_type_ == -1 || activate_type_ == 1))
      << "Build in activate only support Relu";

  if (bias_term_) {
    CHECK_EQ(blobs_size(), 2);
  } else {
    CHECK_EQ(blobs_size(), 1);
  }
}

void DepthwiseConvOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int in_h = bottom->shape(2), in_w = bottom->shape(3);

  VecInt top_shape = bottom->shape();
  top_shape[1] = num_output_;
  top_shape[2] = conv_out_size(in_h, kernel_size_, stride_, pad_, dilation_);
  top_shape[3] = conv_out_size(in_w, kernel_size_, stride_, pad_, dilation_);
  top->reshape(top_shape);

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << num_output_ << "_" << kernel_size_ << "x" << kernel_size_
             << "_s" << stride_ << "_p" << pad_ << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void DepthwiseConvOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bias_term_) {
    Vision::DepthwiseConv(bottom->data(), bottom->shape(),
                          blobs<float>(0)->data(), blobs<float>(1)->data(),
                          kernel_size_, stride_, pad_, bias_term_, top->shape(),
                          top->mutable_data());
  } else {
    Vision::DepthwiseConv(bottom->data(), bottom->shape(),
                          blobs<float>(0)->data(), blobs<float>(0)->data(),
                          kernel_size_, stride_, pad_, bias_term_, top->shape(),
                          top->mutable_data());
  }

  if (activate_type_ == 1) {
    Vision::Activate(top->mutable_data(), top->count(), activate_type_);
  }
}

void DepthwiseConvOp::Release() {
  // DLOG(INFO) << "Free DepthwiseConvOp!";
}

REGISTER_OPERATOR(DepthwiseConv, DepthwiseConvOp);

}  // namespace Shadow
