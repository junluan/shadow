#include "core/operator.hpp"

#include "kernels/resize.hpp"

namespace Shadow {

class ResizeOp : public Operator {
 public:
  ResizeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    if (has_argument("size")) {
      const auto& size = get_repeated_argument<int>("size");
      CHECK_LE(size.size(), 2);
      if (size.empty()) {
        out_h_ = out_w_ = get_single_argument<int>("size", 0);
      } else if (size.size() == 1) {
        out_h_ = out_w_ = size[0];
      } else {
        out_h_ = size[0], out_w_ = size[1];
      }
    } else {
      out_h_ = get_single_argument<int>("out_h", 0);
      out_w_ = get_single_argument<int>("out_w", 0);
    }
    const auto& scale = get_repeated_argument<float>("scale");
    CHECK_LE(scale.size(), 2);
    if (scale.empty()) {
      scale_h_ = scale_w_ = get_single_argument<float>("scale", 1);
    } else if (scale.size() == 1) {
      scale_h_ = scale_w_ = scale[0];
    } else {
      scale_h_ = scale[0], scale_w_ = scale[1];
    }
    type_ = get_single_argument<int>("type", 1);
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

    int in_h = input->shape(2), in_w = input->shape(3);
    int out_h = out_h_, out_w = out_w_;

    if (inputs.size() > 1) {
      const auto& size = inputs[1];
      auto size_type = size->data_type();
      if (size_type == DataType::kI32) {
        CHECK_EQ(size->num_axes(), 1);
        CHECK_EQ(size->count(), 2);
        VecInt size_data(2, 0);
        size->get_data<int>(size_data.data(), 2);
        out_h = size_data[0], out_w = size_data[1];
      } else if (size_type == DataType::kF32) {
        CHECK_EQ(size->num_axes(), 4);
        out_h = size->shape(2), out_w = size->shape(3);
      }
    }

    if (out_h == 0 || out_w == 0) {
      out_h = static_cast<int>(in_h * scale_h_);
      out_w = static_cast<int>(in_w * scale_w_);
    }

    CHECK_GT(out_h, 0);
    CHECK_GT(out_w, 0);

    auto out_shape = input->shape();
    out_shape[2] = out_h;
    out_shape[3] = out_w;
    output->reshape(out_shape);

    kernel_->Run(input, output, ws_, type_, align_corners_);
  }

 private:
  int out_h_, out_w_, type_;
  float scale_h_, scale_w_;
  bool align_corners_;

  std::shared_ptr<ResizeKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Resize, ResizeOp);

}  // namespace Shadow
