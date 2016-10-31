#define CL_KERNEL_LOOP(globalid, count)  \
  const int globalid = get_global_id(0); \
  if (globalid >= count) return;

__kernel void DataTransform(__global float *in_data, int count, int in_c,
                            int spatial_dim, float scale, int num_mean,
                            __global float *mean_value,
                            __global float *out_data) {
  CL_KERNEL_LOOP(globalid, count)

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

__kernel void Im2Col(__global float *im_data, int offset, int in_c, int in_h,
                     int in_w, int kernel_size, int stride, int pad,
                     int dilation, int out_h, int out_w,
                     __global float *col_data) {
  CL_KERNEL_LOOP(globalid, in_c * out_h * out_w)

  const int h_index = globalid / out_w;
  const int h_col = h_index % out_h;
  const int w_col = globalid % out_w;
  const int c_im = h_index / out_h;
  const int c_col = c_im * kernel_size * kernel_size;
  const int h_offset = h_col * stride - pad;
  const int w_offset = w_col * stride - pad;
  col_data += (c_col * out_h + h_col) * out_w + w_col;
  im_data += offset + (c_im * in_h + h_offset) * in_w + w_offset;
  for (int i = 0; i < kernel_size; ++i) {
    for (int j = 0; j < kernel_size; ++j) {
      int h_im = h_offset + i * dilation;
      int w_im = w_offset + j * dilation;
      *col_data = (h_im >= 0 && w_im >= 0 && h_im < in_h && w_im < in_w)
                      ? im_data[i * dilation * in_w + j * dilation]
                      : 0;
      col_data += out_h * out_w;
    }
  }
}

__kernel void Pooling(__global float *in_data, int batch, int in_c, int in_h,
                      int in_w, int kernel_size, int stride, int pad, int mode,
                      int out_h, int out_w, __global float *out_data) {
  CL_KERNEL_LOOP(globalid, batch * in_c * out_h * out_w)

  int b_out = (globalid / (out_w * out_h * in_c)) % batch;
  int c_out = (globalid / (out_w * out_h)) % in_c;
  int i_out = (globalid / out_w) % out_h;
  int j_out = globalid % out_w;

  int kistart = i_out * stride - pad, kjstart = j_out * stride - pad;
  int kiend = min(kistart + kernel_size, in_h);
  int kjend = min(kjstart + kernel_size, in_w);
  int pool_size = (kiend - kistart) * (kjend - kjstart);
  kistart = max(kistart, 0), kjstart = max(kjstart, 0);
  kiend = min(kiend, in_h), kjend = min(kjend, in_w);

  float max = -FLT_MAX;
  float sum = 0.f;
  for (int ki = kistart; ki < kiend; ++ki) {
    for (int kj = kjstart; kj < kjend; ++kj) {
      int index = kj + in_w * (ki + in_h * (c_out + in_c * b_out));
      float value = in_data[index];
      max = (value > max) ? value : max;
      sum += value;
    }
  }
  if (mode == 0) {
    out_data[globalid] = max;
  } else {
    out_data[globalid] = sum / pool_size;
  }
}

__kernel void Concat(__global float *in_data, int count, int num_concats,
                     int concat_size, int top_concat_axis,
                     int bottom_concat_axis, int offset_concat_axis,
                     __global float *out_data) {
  CL_KERNEL_LOOP(globalid, count)

  int total_concat_size = concat_size * bottom_concat_axis;
  int concat_num = globalid / total_concat_size;
  int concat_index = globalid % total_concat_size;
  int top_index =
      concat_index +
      (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
  out_data[top_index] = in_data[globalid];
}

__kernel void Permute(__global float *in_data, int count, int num_axes,
                      __global int *permute_order, __global int *old_steps,
                      __global int *new_steps, __global float *out_data) {
  CL_KERNEL_LOOP(globalid, count)

  int old_idx = 0;
  int idx = globalid;
  for (int j = 0; j < num_axes; ++j) {
    int order = permute_order[j];
    old_idx += (idx / new_steps[j]) * old_steps[order];
    idx %= new_steps[j];
  }
  out_data[globalid] = in_data[old_idx];
}

inline float ActivateValue(float x, int type) {
  switch (type) {
    case 0:
      return x; /*linear*/
    case 1:
      return x * (x > 0); /*relu*/
    case 2:
      return (x > 0) ? x : .1f * x; /*leaky*/
    default:
      return x;
  }
}

__kernel void Activate(__global float *data, int count, int type) {
  CL_KERNEL_LOOP(globalid, count)

  data[globalid] = ActivateValue(data[globalid], type);
}
