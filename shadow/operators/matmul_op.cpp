#include "core/operator.hpp"

#include "kernels/matmul.hpp"

namespace Shadow {

class MatMulOp : public Operator {
 public:
  MatMulOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    transpose_a_ = get_single_argument<bool>("transpose_a", false);
    transpose_b_ = get_single_argument<bool>("transpose_b", false);

    kernel_ = std::dynamic_pointer_cast<MatMulKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    CHECK_EQ(inputs.size(), 2);

    const auto& input_a = inputs[0];
    const auto& input_b = inputs[1];
    auto& output = outputs[0];

    CHECK_NE(input_a, output);
    CHECK_NE(input_b, output);

    int num_axes_a = input_a->num_axes(), num_axes_b = input_b->num_axes();

    CHECK(num_axes_a >= 2 && num_axes_b >= 2);
    if (num_axes_a == num_axes_b) {
      for (int d = 0; d < num_axes_a - 2; ++d) {
        CHECK_EQ(input_a->shape(d), input_b->shape(d));
      }
    } else {
      CHECK(num_axes_a == 2 || num_axes_b == 2);
    }

    int rows_a = input_a->shape(-2), cols_a = input_a->shape(-1);
    int rows_b = input_b->shape(-2), cols_b = input_b->shape(-1);

    int M = transpose_a_ ? cols_a : rows_a;
    int N = transpose_b_ ? rows_b : cols_b;
    int K = transpose_a_ ? rows_a : cols_a;

    CHECK_EQ(K, transpose_b_ ? cols_b : rows_b);

    if (num_axes_a >= num_axes_b) {
      auto out_shape = input_a->shape();
      out_shape[num_axes_a - 2] = M;
      out_shape[num_axes_a - 1] = N;
      output->reshape(out_shape);
    } else {
      auto out_shape = input_b->shape();
      out_shape[num_axes_b - 2] = M;
      out_shape[num_axes_b - 1] = N;
      output->reshape(out_shape);
    }

    kernel_->Run(input_a, input_b, output, ws_, transpose_a_, transpose_b_);
  }

 private:
  bool transpose_a_, transpose_b_;

  std::shared_ptr<MatMulKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(MatMul, MatMulOp);

}  // namespace Shadow
