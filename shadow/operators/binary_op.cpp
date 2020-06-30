#include "core/operator.hpp"

#include "kernels/binary.hpp"

namespace Shadow {

class BinaryOp : public Operator {
 public:
  BinaryOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", -1);
    CHECK_GE(operation_, 0);
    CHECK_LE(operation_, 6);
    has_scalar_ = has_argument("scalar");
    if (has_scalar_) {
      scalar_value_ = get_single_argument<float>("scalar", 0);
    }

    kernel_ = std::dynamic_pointer_cast<BinaryKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), has_scalar_ ? 1 : 2);

    const auto& input = inputs[0];
    auto& output = outputs[0];

    if (has_scalar_) {
      output->reshape(input->shape());
      kernel_->Run(input, output, ws_, operation_, scalar_value_);
    } else {
      auto& scalar = inputs[1];
      if (input->shape() != scalar->shape()) {
        const auto in_shape = input->shape(), scalar_shape = scalar->shape();
        int num_diff_axes = input->num_axes() - scalar->num_axes();
        if (num_diff_axes > 0) {
          auto padded_scalar_shape = scalar_shape;
          padded_scalar_shape.insert(padded_scalar_shape.begin(),
                                     std::abs(num_diff_axes), 1);
          scalar->set_shape(padded_scalar_shape);
        } else if (num_diff_axes < 0) {
          auto padded_in_shape = in_shape;
          padded_in_shape.insert(padded_in_shape.begin(),
                                 std::abs(num_diff_axes), 1);
          input->set_shape(padded_in_shape);
        }
        CHECK_EQ(input->num_axes(), scalar->num_axes());
        VecInt out_shape;
        for (int n = 0; n < input->num_axes(); ++n) {
          int in_dim = input->shape(n), scalar_dim = scalar->shape(n);
          CHECK(in_dim == scalar_dim || in_dim == 1 || scalar_dim == 1);
          out_shape.push_back(std::max(in_dim, scalar_dim));
        }
        output->reshape(out_shape);
        kernel_->Run(input, scalar, output, ws_, operation_, true);
        input->set_shape(in_shape), scalar->set_shape(scalar_shape);
      } else {
        output->reshape(input->shape());
        kernel_->Run(input, scalar, output, ws_, operation_, false);
      }
    }
  }

 private:
  int operation_;
  float scalar_value_;
  bool has_scalar_;

  std::shared_ptr<BinaryKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(Binary, BinaryOp);

}  // namespace Shadow
