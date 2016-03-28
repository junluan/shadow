#ifndef SHADOW_IMAGE_H
#define SHADOW_IMAGE_H

#ifdef USE_CL
#include "cl.h"
#endif

class Image {
public:
  Image();
  ~Image();

  static void GetFloatData(unsigned char *image, int width, int height,
                           int channel, float *data);
  static void Im2Col(float *im_data, int in_c, int in_h, int in_w, int ksize,
                     int stride, int pad, int out_h, int out_w,
                     float *col_data);

#ifdef USE_CL
  static void CLIm2Col(cl_mem im_data, int offset, int in_c, int in_h, int in_w,
                       int ksize, int stride, int pad, int out_h, int out_w,
                       cl_mem col_data);
#endif

private:
  static float Im2ColGetPixel(float *im_data, int height, int width, int row,
                              int col, int channel, int pad);
};

#endif // SHADOW_IMAGE_H
