#include "depthwise_conv_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

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

REGISTER_OPERATOR(DepthwiseConv, DepthwiseConvOp);

}  // namespace Shadow
