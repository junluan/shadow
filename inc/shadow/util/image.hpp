#ifndef SHADOW_UTIL_IMAGE_HPP
#define SHADOW_UTIL_IMAGE_HPP

#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

namespace Image {

template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, float scale,
                   int num_mean, const T *mean_value, T *out_data);

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, const VecInt &out_shape,
            T *out_data);

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data);

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data);

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data);

template <typename T>
void Activate(T *data, int count, int type);

}  // namespace Image

#endif  // SHADOW_UTIL_IMAGE_HPP
