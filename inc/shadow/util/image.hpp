#ifndef SHADOW_UTIL_IMAGE_HPP
#define SHADOW_UTIL_IMAGE_HPP

class Image {
public:
  static void Im2Col(const float *im_data, int in_c, int in_h, int in_w,
                     int ksize, int stride, int pad, int out_h, int out_w,
                     float *col_data);
  static void Pooling(const float *in_data, int batch, int in_c, int in_h,
                      int in_w, int ksize, int stride, int out_h, int out_w,
                      int mode, float *out_data);
};

#endif // SHADOW_UTIL_IMAGE_HPP
