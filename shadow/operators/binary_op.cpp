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

  void Run() override {
    CHECK_EQ(bottoms_size(), has_scalar_ ? 1 : 2);

    const auto bottom = bottoms(0);
    auto top = tops(0);

    if (has_scalar_) {
      top->reshape(bottom->shape());
      kernel_->Run(bottom, top, ws_, operation_, scalar_value_);
    } else {
      const auto scalar = bottoms(1);
      if (bottom->shape() != scalar->shape()) {
        const auto bottom_shape = bottom->shape(),
                   scalar_shape = scalar->shape();
        int num_diff_axes = bottom->num_axes() - scalar->num_axes();
        if (num_diff_axes > 0) {
          auto padded_scalar_shape = scalar_shape;
          padded_scalar_shape.insert(padded_scalar_shape.begin(),
                                     std::abs(num_diff_axes), 1);
          scalar->set_shape(padded_scalar_shape);
        } else if (num_diff_axes < 0) {
          auto padded_bottom_shape = bottom_shape;
          padded_bottom_shape.insert(padded_bottom_shape.begin(),
                                     std::abs(num_diff_axes), 1);
          bottom->set_shape(padded_bottom_shape);
        }
        CHECK_EQ(bottom->num_axes(), scalar->num_axes());
        VecInt top_shape;
        for (int n = 0; n < bottom->num_axes(); ++n) {
          int bottom_dim = bottom->shape(n), scalar_dim = scalar->shape(n);
          CHECK(bottom_dim == scalar_dim || bottom_dim == 1 || scalar_dim == 1);
          top_shape.push_back(std::max(bottom_dim, scalar_dim));
        }
        top->reshape(top_shape);
        kernel_->Run(bottom, scalar, top, ws_, operation_, true);
        bottom->set_shape(bottom_shape), scalar->set_shape(scalar_shape);
      } else {
        top->reshape(bottom->shape());
        kernel_->Run(bottom, scalar, top, ws_, operation_, false);
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
