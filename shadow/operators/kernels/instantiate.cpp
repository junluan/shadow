#include "eltwise.hpp"
#include "unary.hpp"

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(EltwiseCPU, EltwiseKernelDefault<DeviceType::kCPU>);

REGISTER_OP_KERNEL_DEFAULT(UnaryCPU, UnaryKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
