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
    kernel_size_ = get_single_argument<int>("kernel_size", 0);
    stride_ = get_single_argument<int>("stride", 1);
    pad_ = get_single_argument<int>("pad", 0);
    dilation_ = get_single_argument<int>("dilation", 1);
    group_ = get_single_argument<int>("group", 1);
    CHECK_EQ(num_output_ % group_, 0);
    deform_group_ = get_single_argument<int>("deform_group", 1);
    CHECK_EQ(num_output_ % deform_group_, 0);
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

    int in_c = input->shape(1), in_h = input->shape(2), in_w = input->shape(3);

    CHECK_EQ(in_c % group_, 0);
    CHECK_EQ(offset->shape(1) % deform_group_, 0);

    auto out_shape = input->shape();
    out_shape[1] = num_output_;
    out_shape[2] =
        deform_conv_out_size(in_h, kernel_size_, stride_, pad_, dilation_);
    out_shape[3] =
        deform_conv_out_size(in_w, kernel_size_, stride_, pad_, dilation_);
    output->reshape(out_shape);

    kernel_->Run(input, offset, weight,
                 bias_term_ ? inputs[3] : std::shared_ptr<Blob>(nullptr),
                 output, ws_, num_output_, kernel_size_, stride_, pad_,
                 dilation_, group_, deform_group_, bias_term_, activate_type_);
  }

 private:
  int num_output_, kernel_size_, stride_, pad_, dilation_, group_,
      deform_group_, activate_type_;
  bool bias_term_;

  std::shared_ptr<DeformConvKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(DeformConv, DeformConvOp);

}  // namespace Shadow
