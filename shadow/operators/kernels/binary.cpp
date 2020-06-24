#include "binary.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

inline float Binary(float a, float b, int operation) {
  switch (operation) {
    case kAdd:
      return a + b;
    case kSub:
      return a - b;
    case kMul:
      return a * b;
    case kDiv:
      return a / b;
    case kPow:
      return std::pow(a, b);
    case kMax:
      return std::max(a, b);
    case kMin:
      return std::min(a, b);
    default:
      return 0;
  }
}

template <>
void BroadcastBinary<DeviceType::kCPU, float>(
    const float* in_data, const int* in_shape, const float* scalar_data,
    const int* scalar_shape, int operation, int num_axes, int count,
    const int* out_shape, float* out_data, Context* context) {
  VecInt in_shape_acc(1, 1), scalar_shape_acc(1, 1);
  for (int n = num_axes - 1; n > 0; --n) {
    in_shape_acc.insert(in_shape_acc.begin(), in_shape[n] * in_shape_acc[0]);
    scalar_shape_acc.insert(scalar_shape_acc.begin(),
                            scalar_shape[n] * scalar_shape_acc[0]);
  }
  for (int i = 0; i < count; ++i) {
    int in_index = 0, scalar_index = 0, cc = i;
    for (int n = num_axes - 1; n >= 0; --n) {
      int dim = cc % out_shape[n];
      in_index += (dim % in_shape[n]) * in_shape_acc[n];
      scalar_index += (dim % scalar_shape[n]) * scalar_shape_acc[n];
      cc /= out_shape[n];
    }
    out_data[i] =
        Binary(in_data[in_index], scalar_data[scalar_index], operation);
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(BinaryCPU, BinaryKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
