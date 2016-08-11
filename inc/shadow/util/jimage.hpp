#ifndef SHADOW_UTIL_JIMAGE_HPP
#define SHADOW_UTIL_JIMAGE_HPP

#include "shadow/util/util.hpp"

#if defined(USE_OpenCV)
#include <opencv2/opencv.hpp>
#endif

#define USE_ArcSoft
#if defined(USE_ArcSoft)
#include "arcsoft/asvloffscreen.h"
#endif

class Scalar {
 public:
  Scalar() {}
  Scalar(int r_t, int g_t, int b_t) {
    r = (unsigned char)Util::constrain(0, 255, r_t);
    g = (unsigned char)Util::constrain(0, 255, g_t);
    b = (unsigned char)Util::constrain(0, 255, b_t);
  }
  unsigned char r, g, b;
};

enum Order { kGray, kRGB, kBGR, kArc };

class JImage {
 public:
  JImage() : data_(nullptr), c_(0), h_(0), w_(0) {
#if defined(USE_ArcSoft)
    arc_data_ = nullptr;
#endif
  }
  explicit JImage(const std::string &im_path) : data_(nullptr) {
#if defined(USE_ArcSoft)
    arc_data_ = nullptr;
#endif
    Read(im_path);
  }
  JImage(int channel, int height, int width, Order order = kRGB)
      : c_(channel), h_(height), w_(width), order_(order) {
#if defined(USE_ArcSoft)
    arc_data_ = nullptr;
#endif
    data_ = new unsigned char[c_ * h_ * w_];
  }
  ~JImage() { Release(); }

  const unsigned char *data() const { return data_; }
  unsigned char *data() { return data_; }
  Order order() const { return order_; }

  const unsigned char operator()(int c, int h, int w) const {
    if (c >= c_ || h >= h_ || w >= w_) Fatal("Index out of range!");
    return data_[(c * h_ + h) * w_ + w];
  }
  unsigned char &operator()(int c, int h, int w) {
    if (c >= c_ || h >= h_ || w >= w_) Fatal("Index out of range!");
    return data_[(c * h_ + h) * w_ + w];
  }

  void SetData(const unsigned char *data, int count) {
    if (data_ == nullptr) Fatal("JImage data is NULL!");
    if (count > c_ * h_ * w_) Fatal("Set data region overflow!");
    memcpy(data_, data, sizeof(unsigned char) * count);
  }
  void SetZero() { memset(data_, 0, sizeof(unsigned char) * c_ * h_ * w_); }

  void Read(const std::string &im_path);
  void Write(const std::string &im_path);
  void Show(const std::string &show_name, int wait_time = 0);
  void CopyTo(JImage *im_copy) const;
  void Resize(JImage *im_res, int height, int width);
  void Crop(JImage *im_crop, const RectF &crop);
  void CropWithResize(JImage *im_res, const RectF &crop, int height, int width);

  void Color2Gray();

#if defined(USE_OpenCV)
  void FromMat(const cv::Mat &im_mat);
  void FromMatWithCropResize(const cv::Mat &im_mat, const RectF &crop,
                             int resize_h, int resize_w, float *batch_data);
#endif

#if defined(USE_ArcSoft)
  void FromArcImage(const ASVLOFFSCREEN &im_arc);
  void FromArcImageWithCropResize(const ASVLOFFSCREEN &im_arc,
                                  const RectF &crop, int resize_h, int resize_w,
                                  float *batch_data);
  void JImageToArcImage(int arc_format);
#endif

  void Rectangle(const RectI &rect, const Scalar &scalar = Scalar(0, 255, 0));
  void Rectangle(const VecRectI &rects,
                 const Scalar &scalar = Scalar(0, 255, 0));

  VecPoint2i GetNoneZeroPoints(int threshold = 0) const;
  void GetBatchData(float *batch_data);

  void Release();

  int c_, h_, w_;

#if defined(USE_ArcSoft)
  ASVLOFFSCREEN arc_image_;
  unsigned char *arc_data_;
#endif

 private:
  void GetInv(unsigned char *im_inv);

  unsigned char *data_;
  Order order_;
};

#endif  // SHADOW_UTIL_JIMAGE_HPP
