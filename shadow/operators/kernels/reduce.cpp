#include "reduce.hpp"

namespace Shadow {

namespace Vision {

inline float Reduce(const float* data, const int* list, int num_list,
                    int offset, int operation) {
  switch (operation) {
    case kProd: {
      double val = 1;
      for (int i = 0; i < num_list; ++i) {
        val *= data[list[i] + offset];
      }
      return static_cast<float>(val);
    }
    case kSum: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += data[list[i] + offset];
      }
      return static_cast<float>(val);
    }
    case kMax: {
      float val = std::numeric_limits<float>::lowest();
      for (int i = 0; i < num_list; ++i) {
        val = std::max(val, data[list[i] + offset]);
      }
      return val;
    }
    case kMin: {
      float val = std::numeric_limits<float>::max();
      for (int i = 0; i < num_list; ++i) {
        val = std::min(val, data[list[i] + offset]);
      }
      return val;
    }
    case kAvg: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += data[list[i] + offset];
      }
      return static_cast<float>(val / num_list);
    }
    default:
      return 0;
  }
}

template <>
void Reduce<DeviceType::kCPU, float>(const float* in_data, const int* list_data,
                                     const int* offset_data, int num_list,
                                     int operation, int count, float* out_data,
                                     Context* context) {
  for (int i = 0; i < count; ++i) {
    out_data[i] =
        Reduce(in_data, list_data, num_list, offset_data[i], operation);
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ReduceCPU, ReduceKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
