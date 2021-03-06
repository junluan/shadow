#include "core/operator.hpp"

#include "kernels/deform_conv.hpp"

namespace Shadow {

inline int deform_conv_out_size(int dim, int kernel_size, int stride, int pad,
                                int dilation) {
  return (dim + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
}

inline VecInt expand_param(const VecInt& param, int num) {
  if (param.size() == 1) {
    return VecInt(num, param[0]);
  } else {
    CHECK_EQ(param.size(), num);
    return param;
  }
}

class DeformConvOp : public Operator {
 public:
  DeformConvOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    num_output_ = get_single_argument<int>("num_output", 0);
    CHECK(has_argument("kernel_size"));
    kernel_size_ = get_repeated_argument<int>("kernel_size", 0);
    stride_ = get_repeated_argument<int>("stride", 1);
    pad_ = get_repeated_argument<int>("pad", 0);
    dilation_ = get_repeated_argument<int>("dilation", 1);
    group_ = get_single_argument<int>("group", 1);
    CHECK_EQ(num_output_ % group_, 0);
    deform_group_ = get_single_argument<int>("deform_group", 1);
    use_mask_ = get_single_argument<bool>("use_mask", false);
    bias_term_ = get_single_argument<bool>("bias_term", true);
    activate_type_ = get_single_argument<int>("type", -1);
    CHECK(activate_type_ == -1 || activate_type_ == 1)
        << "Build in activate only support Relu";

    kernel_ = std::dynamic_pointer_cast<DeformConvKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    if (use_mask_ && bias_term_) {
      CHECK_EQ(inputs.size(), 5);
    } else if (use_mask_ || bias_term_) {
      CHECK_EQ(inputs.size(), 4);
    } else {
      CHECK_EQ(inputs.size(), 3);
    }

    const auto& input = inputs[0];
    const auto& offset = inputs[1];
    const auto& mask = use_mask_ ? inputs[2] : std::shared_ptr<Blob>(nullptr);
    const auto& weight = use_mask_ ? inputs[3] : inputs[2];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int num_spatial_axes = input->num_axes() - 2;

    CHECK_EQ(num_spatial_axes, 2) << "Only support 2D DeformConv";

    CHECK_EQ(input->shape(0), offset->shape(0));
    CHECK_EQ(input->shape(1) % group_, 0);
    CHECK_EQ(input->shape(1) % deform_group_, 0);
    CHECK_EQ(offset->shape(1), 2 * deform_group_ * weight->count(2));
    if (use_mask_) {
      CHECK_EQ(mask->shape(1), deform_group_ * weight->count(2));
    }

    const auto& kernel_size = expand_param(kernel_size_, num_spatial_axes);
    const auto& stride = expand_param(stride_, num_spatial_axes);
    const auto& pad = expand_param(pad_, num_spatial_axes);
    const auto& dilation = expand_param(dilation_, num_spatial_axes);

    auto out_shape = input->shape();
    out_shape[1] = num_output_;
    for (int n = 0; n < num_spatial_axes; ++n) {
      out_shape[n + 2] = deform_conv_out_size(
          input->shape(n + 2), kernel_size[n], stride[n], pad[n], dilation[n]);
      CHECK_EQ(offset->shape(n + 2), out_shape[n + 2]);
      if (use_mask_) {
        CHECK_EQ(mask->shape(n + 2), out_shape[n + 2]);
      }
    }
    output->reshape(out_shape);

    kernel_->Run(
        input, offset, mask, weight,
        bias_term_ ? inputs[inputs.size() - 1] : std::shared_ptr<Blob>(nullptr),
        output, ws_, num_output_, kernel_size[0], kernel_size[1], stride[0],
        stride[1], pad[0], pad[1], dilation[0], dilation[1], group_,
        deform_group_, use_mask_, bias_term_, activate_type_);
  }

 private:
  int num_output_, group_, deform_group_, activate_type_;
  bool use_mask_, bias_term_;
  VecInt kernel_size_, stride_, pad_, dilation_;

  std::shared_ptr<DeformConvKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(DeformConv, DeformConvOp);

}  // namespace Shadow
