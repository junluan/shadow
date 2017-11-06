#ifndef SHADOW_OPERATORS_DEPTHWISE_CONV_OP_HPP
#define SHADOW_OPERATORS_DEPTHWISE_CONV_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DepthwiseConvOp : public Operator {
 public:
  explicit DepthwiseConvOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
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

  void Reshape() override;
  void Forward() override;

 protected:
  int num_output_, kernel_size_, stride_, pad_, dilation_, group_,
      activate_type_;
  bool bias_term_;
};

static inline int depthwise_conv_out_size(int dim, int kernel_size, int stride,
                                          int pad, int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return (dim + 2 * pad - kernel_extent) / stride + 1;
}

namespace Vision {

template <typename T>
void DepthwiseConv(const T *in_data, const VecInt &in_shape,
                   const T *weight_data, const T *bias_data, int kernel_size,
                   int stride, int pad, int bias_term, const VecInt &out_shape,
                   T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DEPTHWISE_CONV_OP_HPP
