#ifndef SHADOW_OPERATORS_DECONV_OP_HPP
#define SHADOW_OPERATORS_DECONV_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DeconvOp : public Operator {
 public:
  explicit DeconvOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    num_output_ = get_single_argument<int>("num_output", 0);
    CHECK(has_argument("kernel_size"));
    kernel_size_ = get_single_argument<int>("kernel_size", 0);
    stride_ = get_single_argument<int>("stride", 1);
    pad_ = get_single_argument<int>("pad", 0);
    dilation_ = get_single_argument<int>("dilation", 1);
    group_ = get_single_argument<int>("group", 1);
    CHECK_EQ(num_output_ % group_, 0);
    bias_term_ = get_single_argument<bool>("bias_term", true);
    activate_type_ = get_single_argument<int>("type", -1);
    CHECK((activate_type_ == -1 || activate_type_ == 1))
        << "Build in activate only support Relu";
  }

  void Forward() override;

 private:
  int num_output_, kernel_size_, stride_, pad_, dilation_, group_,
      activate_type_, conv_out_spatial_dim_, out_spatial_dim_, kernel_dim_;
  int conv_in_c, conv_out_c, weight_offset_, col_offset_, output_offset_;
  bool bias_term_, use_cudnn_ = false, use_nnpack_ = false;

  BlobF *biases_multiplier_ = nullptr, *col_image_ = nullptr;
};

static inline int deconv_out_size(int dim, int kernel_size, int stride, int pad,
                                  int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return stride * (dim - 1) + kernel_extent - 2 * pad;
}

namespace Vision {

template <typename T>
void Col2Im(const T *col_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation,
            const VecInt &out_shape, T *in_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DECONV_OP_HPP
