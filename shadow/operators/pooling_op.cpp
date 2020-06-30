#include "core/operator.hpp"

#include "kernels/pooling.hpp"

namespace Shadow {

inline int pooling_out_size(int dim, int kernel_size, int stride, int pad,
                            bool full_pooling) {
  if (full_pooling) {
    return static_cast<int>(
        std::ceil((dim + 2.f * pad - kernel_size) / stride) + 1);
  } else {
    return static_cast<int>(
        std::floor((dim + 2.f * pad - kernel_size) / stride) + 1);
  }
}

class PoolingOp : public Operator {
 public:
  PoolingOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    // Max: 0, Avg: 1
    pool_type_ = get_single_argument<int>("pool", 0);
    global_pooling_ = get_single_argument<bool>("global_pooling", false);
    if (!global_pooling_) {
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
    }
    full_pooling_ = get_single_argument<bool>("full_pooling", true);

    kernel_ = std::dynamic_pointer_cast<PoolingKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int in_h = input->shape(2), in_w = input->shape(3);

    if (global_pooling_) {
      kernel_size_h_ = in_h, kernel_size_w_ = in_w;
      stride_h_ = stride_w_ = 1;
      pad_h_ = pad_w_ = 0;
    }

    int out_h = pooling_out_size(in_h, kernel_size_h_, stride_h_, pad_h_,
                                 full_pooling_);
    int out_w = pooling_out_size(in_w, kernel_size_w_, stride_w_, pad_w_,
                                 full_pooling_);
    if (pad_h_) {
      if ((out_h - 1) * stride_h_ >= in_h + pad_h_) out_h--;
    }
    if (pad_w_) {
      if ((out_w - 1) * stride_w_ >= in_w + pad_w_) out_w--;
    }

    auto out_shape = input->shape();
    out_shape[2] = out_h;
    out_shape[3] = out_w;
    output->reshape(out_shape);

    kernel_->Run(input, output, ws_, pool_type_, kernel_size_h_, kernel_size_w_,
                 stride_h_, stride_w_, pad_h_, pad_w_, full_pooling_);
  }

 private:
  int pool_type_, kernel_size_h_, kernel_size_w_, stride_h_, stride_w_, pad_h_,
      pad_w_;
  bool global_pooling_, full_pooling_;

  std::shared_ptr<PoolingKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Pooling, PoolingOp);

}  // namespace Shadow
