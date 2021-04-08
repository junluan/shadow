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

    int rows_a = input_a->shape(-2), cols_a = input_a->shape(-1);
    int rows_b = input_b->shape(-2), cols_b = input_b->shape(-1);

    int M = transpose_a_ ? cols_a : rows_a;
    int N = transpose_b_ ? rows_b : cols_b;
    int K = transpose_a_ ? rows_a : cols_a;

    CHECK_EQ(K, transpose_b_ ? cols_b : rows_b);

    const auto input_a_shape = input_a->shape(),
               input_b_shape = input_b->shape();
    int num_diff_axes = num_axes_a - num_axes_b;
    if (num_diff_axes > 0) {
      auto padded_shape = input_b_shape;
      padded_shape.insert(padded_shape.begin(), std::abs(num_diff_axes), 1);
      input_b->set_shape(padded_shape);
    } else if (num_diff_axes < 0) {
      auto padded_shape = input_a_shape;
      padded_shape.insert(padded_shape.begin(), std::abs(num_diff_axes), 1);
      input_a->set_shape(padded_shape);
    }
    CHECK_EQ(input_a->num_axes(), input_b->num_axes());

    int num_axes = input_a->num_axes();

    VecInt out_shape(num_axes);
    for (int n = 0; n < num_axes - 2; ++n) {
      int input_a_dim = input_a->shape(n), input_b_dim = input_b->shape(n);
      CHECK(input_a_dim == input_b_dim || input_a_dim == 1 || input_b_dim == 1);
      out_shape[n] = std::max(input_a_dim, input_b_dim);
    }
    out_shape[num_axes - 2] = M, out_shape[num_axes - 1] = N;
    output->reshape(out_shape);

    kernel_->Run(input_a, input_b, output, ws_, transpose_a_, transpose_b_);

    input_a->set_shape(input_a_shape), input_b->set_shape(input_b_shape);
  }

 private:
  bool transpose_a_, transpose_b_;

  std::shared_ptr<MatMulKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(MatMul, MatMulOp);

}  // namespace Shadow
