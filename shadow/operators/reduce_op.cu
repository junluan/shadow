#include "reduce_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__device__ T Reduce(const T *data, const int *list, int num_list, int offset,
                    int operation) {
  switch (operation) {
    case ReduceOp::kProd: {
      T val = 1;
      for (int i = 0; i < num_list; ++i) {
        val *= data[list[i] + offset];
      }
      return val;
    }
    case ReduceOp::kSum: {
      T val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += data[list[i] + offset];
      }
      return val;
    }
    case ReduceOp::kMax: {
      T val = -FLT_MAX;
      for (int i = 0; i < num_list; ++i) {
        val = fmaxf(val, data[list[i] + offset]);
      }
      return val;
    }
    case ReduceOp::kMin: {
      T val = FLT_MAX;
      for (int i = 0; i < num_list; ++i) {
        val = fminf(val, data[list[i] + offset]);
      }
      return val;
    }
    case ReduceOp::kAvg: {
      T val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += data[list[i] + offset];
      }
      return val / num_list;
    }
    default:
      return 0;
  }
}

template <typename T>
__global__ void KernelReduce(const T *in_data, const int *list_data,
                             const int *offset_data, int num_list,
                             int operation, int count, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] =
        Reduce(in_data, list_data, num_list, offset_data[globalid], operation);
  }
}

template <typename T>
void Reduce(const T *in_data, const int *list_data, const int *offset_data,
            int num_list, int operation, int count, T *out_data) {
  KernelReduce<T><<<GetBlocks(count), NumThreads>>>(
      in_data, list_data, offset_data, num_list, operation, count, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Reduce(const float *in_data, const int *list_data,
                     const int *offset_data, int num_list, int operation,
                     int count, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
