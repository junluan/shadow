#include "jimage.hpp"
#include "util.hpp"

//#define USE_STB
#if defined(USE_STB)
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#endif

namespace Shadow {

void JImage::Read(const std::string &im_path) {
  CHECK(Path(im_path).is_file()) << "Can not find " << im_path;
  if (data_ != nullptr) {
    delete[] data_;
    data_ = nullptr;
  }
#if defined(USE_OpenCV)
  FromMat(cv::imread(im_path));

#elif defined(USE_STB)
  data_ = stbi_load(im_path.c_str(), &w_, &h_, &c_, 3);
  CHECK_NOTNULL(data_);
  order_ = kRGB;

#else
  LOG(FATAL) << "Not compiled with either OpenCV or STB, could not read image "
             << im_path;
#endif
}

void JImage::Write(const std::string &im_path) const {
  CHECK_NOTNULL(data_);

#if defined(USE_OpenCV)
  cv::imwrite(im_path, ToMat());

#elif defined(USE_STB)
  int is_ok = -1;
  int step = w_ * c_;
  const auto &path = Util::change_extension(im_path, ".png");
  if (order_ == kRGB) {
    is_ok = stbi_write_png(path.c_str(), w_, h_, c_, data_, step);
  } else if (order_ == kBGR) {
    unsigned char *data_inv = new unsigned char[c_ * h_ * w_];
    GetInv(data_inv);
    is_ok = stbi_write_png(path.c_str(), w_, h_, c_, data_inv, step);
    delete[] data_inv;
  } else {
    LOG(FATAL) << "Unsupported format to disk!";
  }
  CHECK(is_ok) << "Failed to write image to " + im_path;

#else
  LOG(FATAL) << "Not compiled with either OpenCV or STB, could not write image "
             << im_path;
#endif
}

void JImage::Show(const std::string &show_name, int wait_time) const {
  CHECK_NOTNULL(data_);

#if defined(USE_OpenCV)
  cv::namedWindow(show_name, cv::WINDOW_NORMAL);
  cv::imshow(show_name, ToMat());
  cv::waitKey(wait_time);

#else
  LOG(WARNING) << "Not compiled with OpenCV, saving image to " << show_name
               << ".png";
  Write(show_name + ".png");
#endif
}

void JImage::CopyTo(JImage *im_copy) const {
  CHECK_NOTNULL(data_);

  im_copy->Reshape(c_, h_, w_, order_);
  memcpy(im_copy->data_, data_, c_ * h_ * w_ * sizeof(unsigned char));
}

#if defined(USE_OpenCV)
void JImage::FromMat(const cv::Mat &im_mat, bool shared) {
  CHECK(!im_mat.empty()) << "Mat data is empty!";

  if (shared) {
    ShareData(im_mat.data);
    c_ = im_mat.channels(), h_ = im_mat.rows, w_ = im_mat.cols, order_ = kBGR;
  } else {
    Reshape(im_mat.channels(), im_mat.rows, im_mat.cols, kBGR);
    memcpy(data_, im_mat.data, c_ * h_ * w_ * sizeof(unsigned char));
  }
}

cv::Mat JImage::ToMat() const {
  if (order_ == kGray) {
    return cv::Mat(h_, w_, CV_8UC1, data_);
  } else if (order_ == kRGB) {
    cv::Mat im_rgb(h_, w_, CV_8UC3, data_), im_bgr;
    cv::cvtColor(im_rgb, im_bgr, CV_RGB2BGR);
    return im_bgr;
  } else if (order_ == kBGR) {
    return cv::Mat(h_, w_, CV_8UC3, data_);
  } else {
    LOG(FATAL) << "Unsupported format to convert to cv mat!";
  }
  return cv::Mat();
}
#endif

void JImage::Release() {
  if (data_ != nullptr && !shared_) {
    delete[] data_;
    data_ = nullptr;
  }
}

void JImage::GetInv(unsigned char *im_inv) const {
  CHECK_NOTNULL(data_);

  if (order_ == kRGB || order_ == kBGR) {
    int spatial_dim = h_ * w_;
    unsigned char *data_index = data_;
    for (int i = 0; i < spatial_dim; ++i) {
      *(im_inv++) = *(data_index + c_ - 1);
      *(im_inv++) = *(data_index + 1);
      *(im_inv++) = *data_index;
      data_index += c_;
    }
  } else {
    LOG(FATAL) << "Unsupported format to get inverse!";
  }
}

}  // namespace Shadow
