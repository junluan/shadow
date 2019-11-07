#include "gather_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelGather(const T *in_data, const int *indexes_data,
                             int num_indexes, int gather_dim, int inner_num,
                             int count, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int gather_num = num_indexes * inner_num;
    int gather_index = indexes_data[(globalid / inner_num) % num_indexes];
    int in_index =
        (gather_index + globalid / gather_num * gather_dim) * inner_num +
        globalid % inner_num;
    out_data[globalid] = in_data[in_index];
  }
}

template <typename T>
void Gather(const T *in_data, const int *indexes_data, int num_indexes,
            int gather_dim, int inner_num, int count, T *out_data) {
  KernelGather<T><<<GetBlocks(count), NumThreads>>>(in_data, indexes_data,
                                                    num_indexes, gather_dim,
                                                    inner_num, count, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Gather(const float *, const int *, int, int, int, int, float *);
#endif

}  // namespace Vision

}  // namespace Shadow
