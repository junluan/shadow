#include "kernel.hpp"
#include "util/log.hpp"
#include "vision.hpp"

#include "math_functions.h"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelDataTransform(const T *in_data, int count, int in_c,
                                    int spatial_dim, float scale, int num_mean,
                                    const T *mean_value, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int c_out = (globalid / spatial_dim) % in_c;
    int s_out = globalid % spatial_dim;

    if (num_mean == 1) {
      out_data[globalid] = (in_data[globalid] - mean_value[0]) * scale;
    } else if (num_mean == in_c) {
      out_data[globalid] = (in_data[globalid] - mean_value[c_out]) * scale;
    } else if (num_mean == in_c * spatial_dim) {
      out_data[globalid] =
          (in_data[globalid] - mean_value[c_out * spatial_dim + s_out]) * scale;
    }
  }
}

template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, float scale,
                   int num_mean, const T *mean_value, T *out_data) {
  int in_c = in_shape[1], spatial_dim = in_shape[2] * in_shape[3];
  int count = in_shape[0] * in_c * spatial_dim;
  KernelDataTransform<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, spatial_dim, scale, num_mean, mean_value, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelIm2Col(const T *in_data, int offset, int count, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int pad, int dilation, int zero_point, int out_h,
                             int out_w, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int h_index = globalid / out_w;
    int h_col = h_index % out_h;
    int w_col = globalid % out_w;
    int c_im = h_index / out_h;
    int c_col = c_im * kernel_size * kernel_size;
    int h_offset = h_col * stride - pad;
    int w_offset = w_col * stride - pad;
    out_data += (c_col * out_h + h_col) * out_w + w_col;
    in_data += offset + (c_im * in_h + h_offset) * in_w + w_offset;
    for (int i = 0; i < kernel_size; ++i) {
      for (int j = 0; j < kernel_size; ++j) {
        int h_im = h_offset + i * dilation;
        int w_im = w_offset + j * dilation;
        *out_data = (h_im >= 0 && w_im >= 0 && h_im < in_h && w_im < in_w)
                        ? in_data[i * dilation * in_w + j * dilation]
                        : static_cast<T>(zero_point);
        out_data += out_h * out_w;
      }
    }
  }
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation, int zero_point,
            const VecInt &out_shape, T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * out_h * out_w;
  KernelIm2Col<T><<<GetBlocks(count), NumThreads>>>(
      in_data, offset, count, in_c, in_h, in_w, kernel_size, stride, pad,
      dilation, zero_point, out_h, out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPooling(const T *in_data, int count, int in_c, int in_h,
                              int in_w, int kernel_size, int stride, int pad,
                              int mode, int out_h, int out_w, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int j_out = globalid % out_w;
    int i_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % in_c;
    int b_out = temp / in_c;

    int kistart = i_out * stride - pad, kjstart = j_out * stride - pad;
    int kiend = min(kistart + kernel_size, in_h + pad);
    int kjend = min(kjstart + kernel_size, in_w + pad);
    int pool_size = (kiend - kistart) * (kjend - kjstart);
    kistart = max(kistart, 0), kjstart = max(kjstart, 0);
    kiend = min(kiend, in_h), kjend = min(kjend, in_w);

    in_data += (b_out * in_c + c_out) * in_h * in_w;

    T max_val = -FLT_MAX, sum_val = T(0);
    for (int ki = kistart; ki < kiend; ++ki) {
      for (int kj = kjstart; kj < kjend; ++kj) {
        T value = in_data[ki * in_w + kj];
        max_val = max(max_val, value);
        sum_val += value;
      }
    }
    out_data[globalid] = (mode == 0) ? max_val : sum_val / pool_size;
  }
}

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelPooling<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, in_h, in_w, kernel_size, stride, pad, mode, out_h,
      out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelConcat(const T *in_data, int count, int num_concats,
                             int concat_size, int top_concat_axis,
                             int bottom_concat_axis, int offset_concat_axis,
                             T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int total_concat_size = concat_size * bottom_concat_axis;
    int concat_num = globalid / total_concat_size;
    int concat_index = globalid % total_concat_size;
    int top_index =
        concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    out_data[top_index] = in_data[globalid];
  }
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  KernelConcat<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, num_concats, concat_size, top_concat_axis,
      bottom_concat_axis, offset_concat_axis, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPermute(const T *in_data, int count, int num_axes,
                              const int *permute_order, const int *old_steps,
                              const int *new_steps, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int old_idx = 0;
    int idx = globalid;
    for (int j = 0; j < num_axes; ++j) {
      int order = permute_order[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    out_data[globalid] = in_data[old_idx];
  }
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  KernelPermute<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, num_axes, permute_order, old_steps, new_steps, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelScale(const T *in_data, int count, const T *scale_data,
                            const T *bias_data, int scale_dim, int inner_dim,
                            T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_dim) % scale_dim;
    out_data[globalid] =
        in_data[globalid] * scale_data[index] + bias_data[index];
  }
}

template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data) {
  KernelScale<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, scale_data, bias_data, scale_dim, inner_dim, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelBias(const T *in_data, int count, const T *bias_data,
                           int bias_dim, int inner_dim, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_dim) % bias_dim;
    out_data[globalid] = in_data[globalid] + bias_data[index];
  }
}

template <typename T>
void Bias(const T *in_data, int count, const T *bias_data, int bias_dim,
          int inner_dim, T *out_data) {
  KernelBias<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, bias_data, bias_dim, inner_dim, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelReorg(const T *in_data, int count, int in_c, int in_h,
                            int in_w, int out_c, int out_h, int out_w,
                            int stride, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w = globalid % out_w;
    int h = temp % out_h;
    temp = temp / out_h;
    int c = temp % out_c;
    int b = temp / out_c;

    int c_in = c % in_c;
    int area = c / in_c;
    int h_in = h * stride + area / stride;
    int w_in = w * stride + area % stride;
    int in_index = ((b * in_c + c_in) * in_h + h_in) * in_w + w_in;
    int out_index = ((b * out_c + c) * out_h + h) * out_w + w;
    out_data[out_index] = in_data[in_index];
  }
}

template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_c = in_c * stride * stride;
  int out_h = in_h / stride, out_w = in_w / stride;
  int count = batch * out_c * out_h * out_w;
  KernelReorg<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, in_h, in_w, out_c, out_h, out_w, stride, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelLRNFillScale(const T *in_data, int count, int in_c,
                                   int in_h, int in_w, int size,
                                   float alpha_over_size, float k,
                                   T *scale_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / in_w;
    int w = globalid % in_w;
    int h = temp % in_h;
    int b = temp / in_h;

    int offset = (b * in_c * in_h + h) * in_w + w, head = 0;
    const T *in_off = in_data + offset;
    T *scale_off = scale_data + offset;
    T accum_scale = T(0);
    int step = in_h * in_w;
    int pre_pad = (size - 1) / 2, post_pad = size - pre_pad - 1;
    while (head < post_pad && head < in_c) {
      accum_scale += in_off[head * step] * in_off[head * step];
      head++;
    }
    while (head < in_c) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -=
            in_off[(head - size) * step] * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      head++;
    }
    while (head < in_c + post_pad) {
      if (head - size >= 0) {
        accum_scale -=
            in_off[(head - size) * step] * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      head++;
    }
  }
}

template <typename T>
__global__ void KernelLRN(const T *in_data, int count, const T *scale_data,
                          float negative_beta, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] =
        in_data[globalid] * pow(scale_data[globalid], negative_beta);
  }
}

template <typename T>
void LRN(const T *in_data, const VecInt &in_shape, int size, float alpha,
         float beta, float k, T *scale_data, T *out_data) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int count = batch * in_h * in_w;
  KernelLRNFillScale<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, in_h, in_w, size, alpha / size, k, scale_data);
  CUDA_CHECK(cudaPeekAtLastError());
  count *= in_c;
  KernelLRN<T><<<GetBlocks(count), NumThreads>>>(in_data, count, scale_data,
                                                 -beta, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPOIPooling(const T *in_data, int count, const T *roi_data,
                                 int in_c, int in_h, int in_w, int pooled_h,
                                 int pooled_w, float spatial_scale,
                                 T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int pw = globalid % pooled_w;
    int ph = (globalid / pooled_w) % pooled_h;
    int c = (globalid / pooled_w / pooled_h) % in_c;
    int n = globalid / pooled_w / pooled_h / in_c;

    roi_data += n * 5;
    int roi_batch_id = static_cast<int>(roi_data[0]);
    int roi_start_w = static_cast<int>(round(roi_data[1] * spatial_scale));
    int roi_start_h = static_cast<int>(round(roi_data[2] * spatial_scale));
    int roi_end_w = static_cast<int>(round(roi_data[3] * spatial_scale));
    int roi_end_h = static_cast<int>(round(roi_data[4] * spatial_scale));

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    T bin_size_h = roi_height / static_cast<T>(pooled_h);
    T bin_size_w = roi_width / static_cast<T>(pooled_w);

    int hstart = static_cast<int>(floor(ph * bin_size_h));
    int wstart = static_cast<int>(floor(pw * bin_size_w));
    int hend = static_cast<int>(ceil((ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil((pw + 1) * bin_size_w));

    hstart = min(max(hstart + roi_start_h, 0), in_h);
    hend = min(max(hend + roi_start_h, 0), in_h);
    wstart = min(max(wstart + roi_start_w, 0), in_w);
    wend = min(max(wend + roi_start_w, 0), in_w);

    bool is_empty = (hend <= hstart) || (wend <= wstart);

    in_data += (roi_batch_id * in_c + c) * in_h * in_w;

    T max_val = is_empty ? 0 : in_data[hstart * in_w + wstart];
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        max_val = max(max_val, in_data[h * in_w + w]);
      }
    }
    out_data[globalid] = max_val;
  }
}

template <typename T>
void ROIPooling(const T *in_data, const VecInt &in_shape, const T *roi_data,
                int num_rois, int pooled_h, int pooled_w, float spatial_scale,
                T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = num_rois * in_c * pooled_h * pooled_w;
  KernelPOIPooling<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, roi_data, in_c, in_h, in_w, pooled_h, pooled_w,
      spatial_scale, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelProposal(int count, const T *anchor_data,
                               const T *score_data, const T *delta_data,
                               const T *info_data, int in_h, int in_w,
                               int num_anchors, int feat_stride, int min_size,
                               T *proposal_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n_out = globalid % num_anchors;
    int w_out = (globalid / num_anchors) % in_w;
    int h_out = globalid / num_anchors / in_w;

    int spatial_dim = in_h * in_w;
    int spatial_offset = h_out * in_w + w_out;
    int delta_offset = n_out * 4 * spatial_dim + spatial_offset;
    T min_box_size = min_size * info_data[2];

    anchor_data += n_out * 4;
    proposal_data += globalid * 6;

    T score = score_data[(num_anchors + n_out) * spatial_dim + spatial_offset];

    T anchor_x = anchor_data[0] + w_out * feat_stride;
    T anchor_y = anchor_data[1] + h_out * feat_stride;
    T anchor_w = anchor_data[2] - anchor_data[0] + 1;
    T anchor_h = anchor_data[3] - anchor_data[1] + 1;
    T anchor_cx = anchor_x + anchor_w * T(0.5);
    T anchor_cy = anchor_y + anchor_h * T(0.5);

    T dx = delta_data[delta_offset];
    T dy = delta_data[delta_offset + spatial_dim];
    T dw = delta_data[delta_offset + spatial_dim * 2];
    T dh = delta_data[delta_offset + spatial_dim * 3];

    T pb_cx = anchor_cx + anchor_w * dx, pb_cy = anchor_cy + anchor_h * dy;
    T pb_w = anchor_w * std::exp(dw), pb_h = anchor_h * std::exp(dh);

    T pb_xmin = pb_cx - pb_w * T(0.5);
    T pb_ymin = pb_cy - pb_h * T(0.5);
    T pb_xmax = pb_cx + pb_w * T(0.5);
    T pb_ymax = pb_cy + pb_h * T(0.5);

    proposal_data[0] = min(max(pb_xmin, T(0)), info_data[1] - 1);
    proposal_data[1] = min(max(pb_ymin, T(0)), info_data[0] - 1);
    proposal_data[2] = min(max(pb_xmax, T(0)), info_data[1] - 1);
    proposal_data[3] = min(max(pb_ymax, T(0)), info_data[0] - 1);
    proposal_data[4] = score;
    pb_w = proposal_data[2] - proposal_data[0] + 1;
    pb_h = proposal_data[3] - proposal_data[1] + 1;
    proposal_data[5] = (pb_w >= min_box_size) && (pb_h >= min_box_size);
  }
}

template <typename T>
void Proposal(const T *anchor_data, const T *score_data, const T *delta_data,
              const T *info_data, const VecInt &in_shape, int num_anchors,
              int feat_stride, int min_size, T *proposal_data) {
  int in_h = in_shape[2], in_w = in_shape[3];
  int count = in_h * in_w * num_anchors;
  KernelProposal<T><<<GetBlocks(count), NumThreads>>>(
      count, anchor_data, score_data, delta_data, info_data, in_h, in_w,
      num_anchors, feat_stride, min_size, proposal_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__device__ float ActivateValue(T x, int type, float slope) {
  // PRelu: 0, Relu: 1, Leaky: 2, Sigmoid: 3, SoftPlus: 4, Tanh: 5
  switch (type) {
    case 1:
      return x * (x > 0);
    case 2:
      return x > 0 ? x : T(slope * x);
    case 3:
      return 1 / (1 + expf(-x));
    case 4:
      return logf(1 + expf(x));
    case 5: {
      T exp_2x = expf(2 * x);
      return (exp_2x - 1) / (exp_2x + 1);
    }
    default:
      return x;
  }
}

template <typename T>
__global__ void KernelActivate(T *data, int count, int type, float slope) {
  CUDA_KERNEL_LOOP(globalid, count) {
    data[globalid] = ActivateValue(data[globalid], type, slope);
  }
}

template <typename T>
void Activate(T *data, int count, int type, float slope) {
  KernelActivate<T><<<GetBlocks(count), NumThreads>>>(data, count, type, slope);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPRelu(T *data, int count, int channels, int dim,
                            int div_factor, const T *slope_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int c = (globalid / dim) % channels / div_factor;
    T value = data[globalid];
    data[globalid] = value > 0 ? value : value * slope_data[c];
  }
}

template <typename T>
void PRelu(T *data, const VecInt &in_shape, bool channel_shared,
           const T *slope_data) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;
  KernelPRelu<T><<<GetBlocks(count), NumThreads>>>(data, count, channels, dim,
                                                   div_factor, slope_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

// Explicit instantiation
template void DataTransform(const float *in_data, const VecInt &in_shape,
                            float scale, int num_mean, const float *mean_value,
                            float *out_data);
template void Im2Col(const float *in_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     int zero_point, const VecInt &out_shape, float *out_data);
template void Pooling(const float *in_data, const VecInt &in_shape,
                      int kernel_size, int stride, int pad, int mode,
                      const VecInt &out_shape, float *out_data);
template void Concat(const float *in_data, int count, int num_concats,
                     int concat_size, int top_concat_axis,
                     int bottom_concat_axis, int offset_concat_axis,
                     float *out_data);
template void Permute(const float *in_data, int count, int num_axes,
                      const int *permute_order, const int *old_steps,
                      const int *new_steps, float *out_data);
template void Scale(const float *in_data, int count, const float *scale_data,
                    const float *bias_data, int scale_dim, int inner_dim,
                    float *out_data);
template void Bias(const float *in_data, int count, const float *bias_data,
                   int bias_dim, int inner_dim, float *out_data);
template void Reorg(const float *in_data, const VecInt &in_shape, int stride,
                    float *out_data);
template void LRN(const float *in_data, const VecInt &in_shape, int size,
                  float alpha, float beta, float k, float *scale_data,
                  float *out_data);
template void ROIPooling(const float *in_data, const VecInt &in_shape,
                         const float *roi_data, int num_rois, int pooled_h,
                         int pooled_w, float spatial_scale, float *out_data);
template void Proposal(const float *anchor_data, const float *score_data,
                       const float *delta_data, const float *info_data,
                       const VecInt &in_shape, int num_anchors, int feat_stride,
                       int min_size, float *proposal_data);
template void Activate(float *data, int count, int type, float slope);
template void PRelu(float *data, const VecInt &in_shape, bool channel_shared,
                    const float *slope_data);
#endif

}  // namespace Vision

}  // namespace Shadow
