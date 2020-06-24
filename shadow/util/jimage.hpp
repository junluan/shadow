#ifndef SHADOW_UTIL_JIMAGE_HPP_
#define SHADOW_UTIL_JIMAGE_HPP_

#include "log.hpp"

#if defined(USE_OpenCV)
#include <opencv2/opencv.hpp>
#endif

#include <cstring>

namespace Shadow {

enum Order { kGray, kRGB, kBGR, kI420 };

class JImage {
 public:
  explicit JImage(const std::string& im_path) { Read(im_path); }
  explicit JImage(int c = 0, int h = 0, int w = 0, Order order = kRGB)
      : c_(c), h_(h), w_(w), order_(order) {
    if (count() != 0) {
      Reshape(c_, h_, w_, order);
    }
  }
  ~JImage() { Release(); }

  void Read(const std::string& im_path);
  void Write(const std::string& im_path) const;
  void Show(const std::string& show_name, int wait_time = 0) const;
  void CopyTo(JImage* im_copy) const;

  void FromRGBA(const unsigned char* data, int h, int w, Order order = kRGB);
#if defined(USE_OpenCV)
  void FromMat(const cv::Mat& im_mat, bool shared = false);
  cv::Mat ToMat() const;
#endif

  void Release();

  const unsigned char* data() const { return data_; }
  unsigned char* data() { return data_; }

  void SetData(const unsigned char* data, int num) {
    CHECK_NOTNULL(data);
    CHECK_NOTNULL(data_);
    CHECK_EQ(num, count()) << "Set data dimension mismatch!";
    memcpy(data_, data, num * sizeof(unsigned char));
  }

  void SetZero() {
    CHECK_NOTNULL(data_);
    memset(data_, 0, count() * sizeof(unsigned char));
  }

  void ShareData(unsigned char* data) {
    CHECK_NOTNULL(data);
    Release();
    data_ = data;
    shared_ = true;
  }

  void Reshape(int c, int h, int w, Order order, bool shared = false) {
    int num = c * h * w;
    CHECK_GT(num, 0) << "Reshape dimension must be greater than zero!";
    if (!shared) {
      if (data_ == nullptr) {
        data_ = new unsigned char[num];
      } else if (num > count()) {
        delete[] data_;
        data_ = new unsigned char[num];
      }
    } else {
      shared_ = true;
    }
    c_ = c, h_ = h, w_ = w, order_ = order;
  }

  const Order& order() const { return order_; }
  Order& order() { return order_; }

  int count() const { return c_ * h_ * w_; }

  const unsigned char operator()(int c, int h, int w) const {
    if (c >= c_ || h >= h_ || w >= w_) LOG(FATAL) << "Index out of range!";
    return data_[(c * h_ + h) * w_ + w];
  }
  unsigned char& operator()(int c, int h, int w) {
    if (c >= c_ || h >= h_ || w >= w_) LOG(FATAL) << "Index out of range!";
    return data_[(c * h_ + h) * w_ + w];
  }

  int c_ = 0, h_ = 0, w_ = 0;

 private:
  void GetInv(unsigned char* im_inv) const;

  unsigned char* data_ = nullptr;
  Order order_ = kRGB;
  bool shared_ = false;
};

}  // namespace Shadow

#endif  // SHADOW_UTIL_JIMAGE_HPP_
