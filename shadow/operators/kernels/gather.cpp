#include "gather.hpp"

namespace Shadow {

namespace Vision {

template <>
void Gather<DeviceType::kCPU, float>(const float* in_data,
                                     const int* indexes_data, int num_indexes,
                                     int gather_dim, int inner_num, int count,
                                     float* out_data, Context* context) {
  int gather_num = num_indexes * inner_num;
  for (int i = 0; i < count; ++i) {
    int gather_index = indexes_data[(i / inner_num) % num_indexes];
    int in_index = (gather_index + i / gather_num * gather_dim) * inner_num +
                   i % inner_num;
    out_data[i] = in_data[in_index];
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(GatherCPU, GatherKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
