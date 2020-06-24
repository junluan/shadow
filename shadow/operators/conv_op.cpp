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
    const auto& kernel_size = get_repeated_argument<int>("kernel_size");
    CHECK_LE(kernel_size.size(), 2);
    if (kernel_size.empty()) {
      kernel_size_h_ = kernel_size_w_ =
          get_single_argument<int>("kernel_size", 0);
    } else if (kernel_size.size() == 1) {
      kernel_size_h_ = kernel_size_w_ = kernel_size[0];
    } else {
      kernel_size_h_ = kernel_size[0], kernel_size_w_ = kernel_size[1];
    }
    const auto& stride = get_repeated_argument<int>("stride");
    CHECK_LE(stride.size(), 2);
    if (stride.empty()) {
      stride_h_ = stride_w_ = get_single_argument<int>("stride", 1);
    } else if (stride.size() == 1) {
      stride_h_ = stride_w_ = stride[0];
    } else {
      stride_h_ = stride[0], stride_w_ = stride[1];
    }
    const auto& pad = get_repeated_argument<int>("pad");
    CHECK_LE(pad.size(), 2);
    if (pad.empty()) {
      pad_h_ = pad_w_ = get_single_argument<int>("pad", 0);
    } else if (pad.size() == 1) {
      pad_h_ = pad_w_ = pad[0];
    } else {
      pad_h_ = pad[0], pad_w_ = pad[1];
    }
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

  void Forward() override {
    CHECK_EQ(bottoms_size(), bias_term_ ? 3 : 2);

    const auto bottom = bottoms(0);
    auto top = tops(0);

    CHECK_NE(bottom, top);

    int in_c = bottom->shape(1), in_h = bottom->shape(2),
        in_w = bottom->shape(3);

    CHECK_EQ(in_c % group_, 0);

    auto top_shape = bottom->shape();
    top_shape[1] = num_output_;
    top_shape[2] =
        conv_out_size(in_h, kernel_size_h_, stride_h_, pad_h_, dilation_);
    top_shape[3] =
        conv_out_size(in_w, kernel_size_w_, stride_w_, pad_w_, dilation_);
    top->reshape(top_shape);

    kernel_->Run(bottom, bottoms(1), bias_term_ ? bottoms(2) : nullptr, top,
                 ws_, num_output_, kernel_size_h_, kernel_size_w_, stride_h_,
                 stride_w_, pad_h_, pad_w_, dilation_, group_, bias_term_,
                 activate_type_);
  }

 private:
  int num_output_, kernel_size_h_, kernel_size_w_, stride_h_, stride_w_, pad_h_,
      pad_w_, dilation_, group_, activate_type_;
  bool bias_term_;

  std::shared_ptr<ConvKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Conv, ConvOp);

}  // namespace Shadow
