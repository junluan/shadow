#ifndef SHADOW_UTIL_IMAGE_HPP
#define SHADOW_UTIL_IMAGE_HPP

#include "shadow/kernel.hpp"

class Image {
 public:
  static void DataTransform(int N, const BType *in_data, float scale,
                            float mean_value, BType *out_data);
  static void Im2Col(const std::vector<int> &in_shape, const BType *in_data,
                     int offset, int kernel_size, int stride, int pad,
                     const std::vector<int> &out_shape, BType *out_data);
  static void Pooling(const std::vector<int> &in_shape, const BType *in_data,
                      int kernel_size, int stride, int mode,
                      const std::vector<int> &out_shape, BType *out_data);
  static void Permute(const BType *in_data, int count, int num_axes,
                      const std::vector<int> &permute_order,
                      const std::vector<int> &old_steps,
                      const std::vector<int> &new_steps, BType *out_data);
};

#endif  // SHADOW_UTIL_IMAGE_HPP
