#ifndef SHADOW_CORE_VISION_HPP
#define SHADOW_CORE_VISION_HPP

#include "util/type.hpp"

namespace Shadow {

namespace Vision {

template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, float scale,
                   int num_mean, const T *mean_value, T *out_data);

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation, int zero_point,
            const VecInt &out_shape, T *out_data);

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
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data);

template <typename T>
void Bias(const T *in_data, int count, const T *bias_data, int bias_dim,
          int inner_dim, T *out_data);

template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data);

template <typename T>
void LRN(const T *in_data, const VecInt &in_shape, int size, float alpha,
         float beta, float k, T *scale_data, T *out_data);

template <typename T>
void ROIPooling(const T *in_data, const VecInt &in_shape, const T *roi_data,
                int num_rois, int pooled_h, int pooled_w, float spatial_scale,
                T *out_data);

template <typename T>
void Activate(T *data, int count, int type, float slope = 0.1);

template <typename T>
void PRelu(T *data, const VecInt &in_shape, bool channel_shared,
           const T *slope_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_CORE_VISION_HPP
