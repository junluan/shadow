#ifndef SHADOW_OPERATORS_KERNELS_MATMUL_HPP_
#define SHADOW_OPERATORS_KERNELS_MATMUL_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

namespace Shadow {

class MatMulKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input_a,
                   const std::shared_ptr<Blob>& input_b,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   bool transpose_a, bool transpose_b) = 0;
};

template <DeviceType D>
class MatMulKernelDefault : public MatMulKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input_a,
           const std::shared_ptr<Blob>& input_b, std::shared_ptr<Blob>& output,
           Workspace* ws, bool transpose_a, bool transpose_b) override {
    int num_axes_a = input_a->num_axes(), num_axes_b = input_b->num_axes();

    int rows_a = input_a->shape(-2), cols_a = input_a->shape(-1);
    int rows_b = input_b->shape(-2), cols_b = input_b->shape(-1);

    int M = transpose_a ? cols_a : rows_a;
    int N = transpose_b ? rows_b : cols_b;
    int K = transpose_a ? rows_a : cols_a;

    int outer_num = output->count(0, output->num_axes() - 2), inner_num = M * N;
    int inner_num_a = num_axes_a >= num_axes_b ? (rows_a * cols_a) : 0;
    int inner_num_b = num_axes_a <= num_axes_b ? (rows_b * cols_b) : 0;

    for (int n = 0; n < outer_num; ++n) {
      Blas::BlasSgemm<D, float>(
          transpose_a, transpose_b, M, N, K, 1, input_a->data<float>(),
          n * inner_num_a, input_b->data<float>(), n * inner_num_b, 0,
          output->mutable_data<float>(), n * inner_num, ws->Ctx().get());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_MATMUL_HPP_
