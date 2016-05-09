#ifndef SHADOW_JIMAGE_HPP
#define SHADOW_JIMAGE_HPP

#include "boxes.hpp"

#include <string>

#ifdef USE_OpenCV
#include <opencv2/opencv.hpp>
#endif

enum Order { kRGB, kBGR };

class JImage {
public:
  JImage();
  explicit JImage(std::string im_path);
  JImage(int channel, int height, int width, Order order = kRGB);
  ~JImage();

  void Read(std::string im_path);
  void Write(std::string im_path);
  void Show(std::string show_name);
  void CopyTo(JImage *im_copy);
  void Resize(JImage *im_res, int height, int width);
  void Crop(JImage *im_crop, Box crop);
  void Filter2D(float *kernel, int height, int width);

  void FromI420(unsigned char *src_y, unsigned char *src_u,
                unsigned char *src_v, int src_h, int src_w, int src_stride);
  void FromI420WithCropResize(unsigned char *src_y, unsigned char *src_u,
                              unsigned char *src_v, int src_h, int src_w,
                              int src_stride, Box roi, int resize_h,
                              int resize_w, float *batch_data);
#ifdef USE_OpenCV
  void FromMat(const cv::Mat &im_mat);
  void FromMatWithCropResize(const cv::Mat &im_mat, Box roi, int resize_h,
                             int resize_w, float *batch_data);
#endif

  void Rectangle(const VecBox &boxes, bool console_show = true);
  void GetBatchData(float *batch_data);

  void Release();

  unsigned char *data_;
  int c_, h_, w_;
  Order order_;

private:
  void GetInv(unsigned char *im_inv);
};

#endif // SHADOW_JIMAGE_HPP
