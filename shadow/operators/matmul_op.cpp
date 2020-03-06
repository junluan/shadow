#include "matmul_op.hpp"

namespace Shadow {

void MatMulOp::Forward() {
  CHECK_EQ(bottoms_size(), 2);

  const auto *bottom_a = bottoms<float>(0);
  const auto *bottom_b = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

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

  int outer_num = top->count(0, top->num_axes() - 2), inner_num = M * N;
  int inner_num_a = num_axes_a >= num_axes_b ? (rows_a * cols_a) : 0;
  int inner_num_b = num_axes_a <= num_axes_b ? (rows_b * cols_b) : 0;

  for (int n = 0; n < outer_num; ++n) {
    Blas::BlasSgemm(transpose_a_, transpose_b_, M, N, K, 1, bottom_a->data(),
                    n * inner_num_a, bottom_b->data(), n * inner_num_b, 0,
                    top->mutable_data(), n * inner_num, op_ws_->Ctx());
  }
}

REGISTER_OPERATOR(MatMul, MatMulOp);

}  // namespace Shadow
