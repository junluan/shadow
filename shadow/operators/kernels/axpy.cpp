#include "axpy.hpp"

#include "core/blas.hpp"

namespace Shadow {

namespace Vision {

template <>
void Axpy<DeviceType::kCPU, float>(const float* alpha_data, const float* x_data,
                                   const float* y_data, int outer_num,
                                   int inner_num, float* out_data,
                                   Context* context) {
  memcpy(out_data, y_data, outer_num * inner_num * sizeof(float));
  for (int n = 0; n < outer_num; ++n) {
    int data_offset = n * inner_num;
    Blas::BlasSaxpy<DeviceType::kCPU, float>(inner_num, alpha_data[n], x_data,
                                             data_offset, out_data, data_offset,
                                             context);
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(AxpyCPU, AxpyKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
