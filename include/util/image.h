#ifndef SHADOW_IMAGE_H
#define SHADOW_IMAGE_H

class Image {
public:
  static void GetFloatData(unsigned char *image, int width, int height,
                           int channel, float *data);
  static void Im2Col(float *im_data, int in_c, int in_h, int in_w, int ksize,
                     int stride, int pad, int out_h, int out_w,
                     float *col_data);
  static void Pooling(float *in_data, int batch, int in_c, int in_h, int in_w,
                      int ksize, int stride, int out_h, int out_w, int mode,
                      float *out_data);
};

#endif // SHADOW_IMAGE_H
