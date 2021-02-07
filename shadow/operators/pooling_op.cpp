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

inline VecInt expand_param(const VecInt& param, int num) {
  if (param.size() == 1) {
    return VecInt(num, param[0]);
  } else {
    CHECK_EQ(param.size(), num);
    return param;
  }
}

class PoolingOp : public Operator {
 public:
  PoolingOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    // Max: 0, Avg: 1
    pool_type_ = get_single_argument<int>("pool", 0);
    CHECK_GE(pool_type_, 0);
    CHECK_LE(pool_type_, 1);
    global_pooling_ = get_single_argument<bool>("global_pooling", false);
    if (global_pooling_) {
      stride_ = VecInt(1, 1), pad_ = VecInt(1, 0);
    } else {
      kernel_size_ = get_repeated_argument<int>("kernel_size", 0);
      stride_ = get_repeated_argument<int>("stride", 1);
      pad_ = get_repeated_argument<int>("pad", 0);
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

    int num_spatial_axes = input->num_axes() - 2;

    CHECK(num_spatial_axes == 2 || num_spatial_axes == 3)
        << "Only support 2D or 3D Pooling";

    if (global_pooling_ && kernel_size_.size() != num_spatial_axes) {
      for (int n = 0; n < num_spatial_axes; ++n) {
        kernel_size_.push_back(input->shape(n + 2));
      }
    }

    const auto& kernel_size = expand_param(kernel_size_, num_spatial_axes);
    const auto& stride = expand_param(stride_, num_spatial_axes);
    const auto& pad = expand_param(pad_, num_spatial_axes);

    auto out_shape = input->shape();
    for (int n = 0; n < num_spatial_axes; ++n) {
      out_shape[n + 2] = pooling_out_size(input->shape(n + 2), kernel_size[n],
                                          stride[n], pad[n], full_pooling_);
    }
    output->reshape(out_shape);

    if (num_spatial_axes == 2) {
      kernel_->Run(input, output, ws_, pool_type_, kernel_size[0],
                   kernel_size[1], stride[0], stride[1], pad[0], pad[1],
                   full_pooling_);
    } else if (num_spatial_axes == 3) {
      kernel_->Run(input, output, ws_, pool_type_, kernel_size[0],
                   kernel_size[1], kernel_size[2], stride[0], stride[1],
                   stride[2], pad[0], pad[1], pad[2], full_pooling_);
    }
  }

 private:
  int pool_type_;
  bool global_pooling_, full_pooling_;
  VecInt kernel_size_, stride_, pad_;

  std::shared_ptr<PoolingKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Pooling, PoolingOp);

}  // namespace Shadow
