#include "core/operator.hpp"

#include "kernels/resize.hpp"

namespace Shadow {

template <typename T>
inline std::vector<T> expand_param(const std::vector<T>& param, int num) {
  if (param.size() == 1) {
    return std::vector<T>(num, param[0]);
  } else {
    CHECK_EQ(param.size(), num);
    return param;
  }
}

class ResizeOp : public Operator {
 public:
  ResizeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    size_ = get_repeated_argument<int>("size", 0);
    scale_ = get_repeated_argument<float>("scale", 1);
    // Nearest: 0, Bilinear: 1
    type_ = get_single_argument<int>("type", 1);
    CHECK_GE(type_, 0);
    CHECK_LE(type_, 1);
    align_corners_ = get_single_argument<bool>("align_corners", false);

    kernel_ = std::dynamic_pointer_cast<ResizeKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    CHECK_NE(input, output);

    int num_spatial_axes = input->num_axes() - 2;

    CHECK_EQ(num_spatial_axes, 2) << "Only support 2D Resize";

    const auto& size = expand_param(size_, num_spatial_axes);
    const auto& scale = expand_param(scale_, num_spatial_axes);

    auto out_shape = input->shape();
    for (int n = 0; n < num_spatial_axes; ++n) {
      out_shape[n + 2] = size[n] > 0
                             ? size[n]
                             : static_cast<int>(input->shape(n + 2) * scale[n]);
      CHECK_GT(out_shape[n + 2], 0);
    }
    output->reshape(out_shape);

    kernel_->Run(input, output, ws_, type_, align_corners_);
  }

 private:
  int type_;
  bool align_corners_;
  VecInt size_;
  VecFloat scale_;

  std::shared_ptr<ResizeKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Resize, ResizeOp);

}  // namespace Shadow
