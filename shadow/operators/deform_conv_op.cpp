#include "core/operator.hpp"

#include "kernels/deform_conv.hpp"

namespace Shadow {

inline int deform_conv_out_size(int dim, int kernel_size, int stride, int pad,
                                int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return (dim + 2 * pad - kernel_extent) / stride + 1;
}

class DeformConvOp : public Operator {
 public:
  DeformConvOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    num_output_ = get_single_argument<int>("num_output", 0);
    CHECK(has_argument("kernel_size"));
    const auto& kernel_size = get_paired_argument<int>("kernel_size", 0);
    kernel_size_h_ = kernel_size.first, kernel_size_w_ = kernel_size.second;
    const auto& stride = get_paired_argument<int>("stride", 1);
    stride_h_ = stride.first, stride_w_ = stride.second;
    const auto& pad = get_paired_argument<int>("pad", 0);
    pad_h_ = pad.first, pad_w_ = pad.second;
    dilation_ = get_single_argument<int>("dilation", 1);
    group_ = get_single_argument<int>("group", 1);
    CHECK_EQ(num_output_ % group_, 0);
    deform_group_ = get_single_argument<int>("deform_group", 1);
    bias_term_ = get_single_argument<bool>("bias_term", true);
    activate_type_ = get_single_argument<int>("type", -1);
    CHECK((activate_type_ == -1 || activate_type_ == 1))
        << "Build in activate only support Relu";

    kernel_ = std::dynamic_pointer_cast<DeformConvKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), bias_term_ ? 4 : 3);

    const auto& input = inputs[0];
    const auto& offset = inputs[1];
    const auto& weight = inputs[2];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    CHECK_EQ(input->shape(0), offset->shape(0));
    CHECK_EQ(input->shape(1) % group_, 0);
    CHECK_EQ(input->shape(1) % deform_group_, 0);
    CHECK_EQ(offset->shape(1),
             2 * deform_group_ * kernel_size_h_ * kernel_size_w_);

    auto out_shape = input->shape();
    out_shape[1] = num_output_;
    out_shape[2] = deform_conv_out_size(input->shape(2), kernel_size_h_,
                                        stride_h_, pad_h_, dilation_);
    out_shape[3] = deform_conv_out_size(input->shape(3), kernel_size_w_,
                                        stride_w_, pad_w_, dilation_);
    output->reshape(out_shape);

    CHECK_EQ(offset->shape(2), output->shape(2));
    CHECK_EQ(offset->shape(3), output->shape(3));

    kernel_->Run(input, offset, weight,
                 bias_term_ ? inputs[3] : std::shared_ptr<Blob>(nullptr),
                 output, ws_, num_output_, kernel_size_h_, kernel_size_w_,
                 stride_h_, stride_w_, pad_h_, pad_w_, dilation_, group_,
                 deform_group_, bias_term_, activate_type_);
  }

 private:
  int num_output_, kernel_size_h_, kernel_size_w_, stride_h_, stride_w_, pad_h_,
      pad_w_, dilation_, group_, deform_group_, activate_type_;
  bool bias_term_;

  std::shared_ptr<DeformConvKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(DeformConv, DeformConvOp);

}  // namespace Shadow
