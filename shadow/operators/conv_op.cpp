#include "core/operator.hpp"

#include "kernels/conv.hpp"

namespace Shadow {

inline int conv_out_size(int dim, int kernel_size, int stride, int pad,
                         int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return (dim + 2 * pad - kernel_extent) / stride + 1;
}

class ConvOp : public Operator {
 public:
  ConvOp(const shadow::OpParam& op_param, Workspace* ws)
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
    bias_term_ = get_single_argument<bool>("bias_term", true);
    activate_type_ = get_single_argument<int>("type", -1);
    CHECK((activate_type_ == -1 || activate_type_ == 1))
        << "Build in activate only support Relu";

    kernel_ = std::dynamic_pointer_cast<ConvKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), bias_term_ ? 3 : 2);

    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int in_c = input->shape(1), in_h = input->shape(2), in_w = input->shape(3);

    CHECK_EQ(in_c % group_, 0);

    auto out_shape = input->shape();
    out_shape[1] = num_output_;
    out_shape[2] =
        conv_out_size(in_h, kernel_size_h_, stride_h_, pad_h_, dilation_);
    out_shape[3] =
        conv_out_size(in_w, kernel_size_w_, stride_w_, pad_w_, dilation_);
    output->reshape(out_shape);

    kernel_->Run(input, weight,
                 bias_term_ ? inputs[2] : std::shared_ptr<Blob>(nullptr),
                 output, ws_, num_output_, kernel_size_h_, kernel_size_w_,
                 stride_h_, stride_w_, pad_h_, pad_w_, dilation_, group_,
                 bias_term_, activate_type_);
  }

 private:
  int num_output_, kernel_size_h_, kernel_size_w_, stride_h_, stride_w_, pad_h_,
      pad_w_, dilation_, group_, activate_type_;
  bool bias_term_;

  std::shared_ptr<ConvKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Conv, ConvOp);

}  // namespace Shadow
