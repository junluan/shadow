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

enum Order { kGray, kRGB, kBGR, kArc };

class JImage {
 public:
  explicit JImage(const std::string &im_path) : data_(nullptr) {
    Read(im_path);

#if defined(USE_ArcSoft)
    arc_data_ = nullptr;
#endif
  }
  explicit JImage(int c = 0, int h = 0, int w = 0, Order order = kRGB)
      : c_(c), h_(h), w_(w), order_(order), data_(nullptr) {
    if (count() != 0) {
      this->Reshape(c_, h_, w_, order);
    }

#if defined(USE_ArcSoft)
    arc_data_ = nullptr;
#endif
  }
  ~JImage() { Release(); }

  const unsigned char *data() const { return data_; }
  unsigned char *data() { return data_; }
  const Order order() const { return order_; }
  const int count() const { return c_ * h_ * w_; }

  const unsigned char operator()(int c, int h, int w) const {
    if (c >= c_ || h >= h_ || w >= w_) Fatal("Index out of range!");
    return data_[(c * h_ + h) * w_ + w];
  }
  unsigned char &operator()(int c, int h, int w) {
    if (c >= c_ || h >= h_ || w >= w_) Fatal("Index out of range!");
    return data_[(c * h_ + h) * w_ + w];
  }

  inline void Reshape(int c, int h, int w, Order order) {
    int num = c * h * w;
    if (num == 0) Fatal("Reshape dimension must be greater than zero!");
    if (data_ == nullptr) {
      data_ = new unsigned char[num];
    } else if (num > count()) {
      delete[] data_;
      data_ = new unsigned char[num];
    }
    c_ = c, h_ = h, w_ = w, order_ = order;
  }

  void SetData(const unsigned char *data, int num) {
    if (data_ == nullptr) Fatal("JImage data is NULL!");
    if (num != count()) Fatal("Set data dimension mismatch!");
    memcpy(data_, data, sizeof(unsigned char) * num);
  }
  void SetZero() {
    if (data_ == nullptr) Fatal("JImage data is NULL!");
    memset(data_, 0, sizeof(unsigned char) * count());
  }

  void Read(const std::string &im_path);
  void Write(const std::string &im_path) const;
  void Show(const std::string &show_name, int wait_time = 0) const;
  void CopyTo(JImage *im_copy) const;

  void Color2Gray();

#if defined(USE_OpenCV)
  void FromMat(const cv::Mat &im_mat);
  cv::Mat ToMat() const;
#endif

#if defined(USE_ArcSoft)
  void FromArcImage(const ASVLOFFSCREEN &im_arc);
  void ToArcImage(int arc_format);
#endif

  void Release();

  int c_, h_, w_;

#if defined(USE_ArcSoft)
  ASVLOFFSCREEN arc_image_;
  unsigned char *arc_data_;
#endif

 private:
  inline void GetInv(unsigned char *im_inv) const;

  unsigned char *data_;
  Order order_;
};

#endif  // SHADOW_UTIL_JIMAGE_HPP
