#ifndef SHADOW_UTIL_IMAGE_HPP
#define SHADOW_UTIL_IMAGE_HPP

#include "shadow/kernel.hpp"

#include <vector>

namespace Image {

template <typename T>
void DataTransform(int N, const T *in_data, float scale, float mean_value,
                   T *out_data);

template <typename T>
void Im2Col(const std::vector<int> &in_shape, const T *in_data, int offset,
            int kernel_size, int stride, int pad,
            const std::vector<int> &out_shape, T *out_data);

template <typename T>
void Pooling(const std::vector<int> &in_shape, const T *in_data,
             int kernel_size, int stride, int mode,
             const std::vector<int> &out_shape, T *out_data);

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data);

}  // namespace Image

#endif  // SHADOW_UTIL_IMAGE_HPP
