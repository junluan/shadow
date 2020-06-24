#include "connected.hpp"
#include "eltwise.hpp"
#include "matmul.hpp"
#include "unary.hpp"

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ConnectedGPU,
                           ConnectedKernelDefault<DeviceType::kGPU>);

REGISTER_OP_KERNEL_DEFAULT(EltwiseGPU, EltwiseKernelDefault<DeviceType::kGPU>);

REGISTER_OP_KERNEL_DEFAULT(MatMulGPU, MatMulKernelDefault<DeviceType::kGPU>);

REGISTER_OP_KERNEL_DEFAULT(UnaryGPU, UnaryKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
