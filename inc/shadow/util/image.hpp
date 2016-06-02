#ifndef SHADOW_UTIL_IMAGE_HPP
#define SHADOW_UTIL_IMAGE_HPP

#include "shadow/kernel.hpp"

class Image {
public:
  static void DataTransform(int N, const BType *in_data, float scale,
                            float mean_value, BType *out_data);
  static void Im2Col(const BType *im_data, int offset, int in_c, int in_h,
                     int in_w, int ksize, int stride, int pad, int out_h,
                     int out_w, BType *col_data);
  static void Pooling(const BType *in_data, int batch, int in_c, int in_h,
                      int in_w, int ksize, int stride, int out_h, int out_w,
                      int mode, BType *out_data);
};

#endif // SHADOW_UTIL_IMAGE_HPP
