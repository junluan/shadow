#ifndef SHADOW_IMAGE_H
#define SHADOW_IMAGE_H

#include "boxes.h"

#include <string>

#ifdef USE_OpenCV
#include <opencv2/opencv.hpp>
#endif

enum Order { kRGB, kBGR, kFRGB };

struct Scalar {
  unsigned char r, g, b;
};

struct image {
  unsigned char *data = NULL;
  int c, h, w;
  Order order;
};

class Image {
public:
#ifdef USE_OpenCV
  static void MatToImage(cv::Mat &im_mat, image &im);
  static void ImRectangle(cv::Mat &im_mat, VecBox &boxes,
                          bool console_show = true);
#endif
  static image MakeImage(int channel, int height, int width,
                         Order order = kRGB);
  static image ImRead(std::string im_path);
  static void ImWrite(std::string im_path, image im);
  static void ImShow(std::string show_name, image im);
  static void ImResize(image im, image &im_res);
  static void ImRectangle(image &im, VecBox &boxes, bool console_show = true);
  static void ImFlatten(image im, image &im_fl);
  static void ImInverse(image im, image &im_inv);

  static void GenBatchData(image im, float *batch_data);

  static void Im2Col(float *im_data, int in_c, int in_h, int in_w, int ksize,
                     int stride, int pad, int out_h, int out_w,
                     float *col_data);
  static void Pooling(float *in_data, int batch, int in_c, int in_h, int in_w,
                      int ksize, int stride, int out_h, int out_w, int mode,
                      float *out_data);
};

#endif // SHADOW_IMAGE_H
