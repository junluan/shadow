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

  void Run() override {
    CHECK_EQ(bottoms_size(), 2);

    const auto bottom_a = bottoms(0);
    const auto bottom_b = bottoms(1);
    auto top = tops(0);

    CHECK_NE(bottom_a, top);
    CHECK_NE(bottom_b, top);

    int num_axes_a = bottom_a->num_axes(), num_axes_b = bottom_b->num_axes();

    CHECK(num_axes_a >= 2 && num_axes_b >= 2);
    if (num_axes_a == num_axes_b) {
      for (int d = 0; d < num_axes_a - 2; ++d) {
        CHECK_EQ(bottom_a->shape(d), bottom_b->shape(d));
      }
    } else {
      CHECK(num_axes_a == 2 || num_axes_b == 2);
    }

    int rows_a = bottom_a->shape(-2), cols_a = bottom_a->shape(-1);
    int rows_b = bottom_b->shape(-2), cols_b = bottom_b->shape(-1);

    int M = transpose_a_ ? cols_a : rows_a;
    int N = transpose_b_ ? rows_b : cols_b;
    int K = transpose_a_ ? rows_a : cols_a;

    CHECK_EQ(K, transpose_b_ ? cols_b : rows_b);

    if (num_axes_a >= num_axes_b) {
      auto top_shape = bottom_a->shape();
      top_shape[num_axes_a - 2] = M;
      top_shape[num_axes_a - 1] = N;
      top->reshape(top_shape);
    } else {
      auto top_shape = bottom_b->shape();
      top_shape[num_axes_b - 2] = M;
      top_shape[num_axes_b - 1] = N;
      top->reshape(top_shape);
    }

    kernel_->Run(bottom_a, bottom_b, top, ws_, transpose_a_, transpose_b_);
  }

 private:
  bool transpose_a_, transpose_b_;

  std::shared_ptr<MatMulKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(MatMul, MatMulOp);

}  // namespace Shadow
