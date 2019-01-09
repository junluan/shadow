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
    auto data_inv = std::vector<unsigned char>(c_ * h_ * w_, 0);
    GetInv(data_inv.data());
    is_ok = stbi_write_png(path.c_str(), w_, h_, c_, data_inv.data(), step);
  } else {
    LOG(FATAL) << "Unsupported format " << order_ << " to disk!";
  }
  CHECK(is_ok) << "Failed to write image to " + path;

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
  CHECK_NOTNULL(im_copy);
  CHECK_NOTNULL(data_);

  im_copy->Reshape(c_, h_, w_, order_);
  im_copy->SetData(data_, count());
}

void JImage::FromRGBA(const unsigned char *data, int h, int w, Order order) {
  CHECK_NOTNULL(data);

  int loc_r = 0, loc_g = 1, loc_b = 2;
  if (order == kGray) {
    loc_r = loc_g = loc_b = 0;
  } else if (order == kRGB) {
    loc_r = 0, loc_g = 1, loc_b = 2;
  } else if (order == kBGR) {
    loc_r = 2, loc_g = 1, loc_b = 0;
  } else {
    LOG(FATAL) << "Unsupported format " << order;
  }

  int spatial_dim = h * w;
  if (order == kGray) {
    Reshape(1, h, w, order);
    unsigned char *data_index = data_;
    for (int i = 0; i < spatial_dim; ++i, data_index += c_, data += 4) {
      auto val = static_cast<int>(0.299f * data[0] + 0.587f * data[1] +
                                  0.114f * data[2]);
      *data_index = (unsigned char)Util::constrain(0, 255, val);
    }
  } else if (order == kRGB || order == kBGR) {
    Reshape(3, h, w, order);
    unsigned char *data_index = data_;
    for (int i = 0; i < spatial_dim; ++i, data_index += c_, data += 4) {
      data_index[loc_r] = data[0];
      data_index[loc_g] = data[1];
      data_index[loc_b] = data[2];
    }
  } else {
    LOG(FATAL) << "Unsupported format " << order_;
  }
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
    cv::cvtColor(im_rgb, im_bgr, cv::COLOR_RGB2BGR);
    return im_bgr;
  } else if (order_ == kBGR) {
    return cv::Mat(h_, w_, CV_8UC3, data_);
  } else {
    LOG(FATAL) << "Unsupported format " << order_ << " to convert to cv mat!";
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
    for (int i = 0; i < spatial_dim; ++i, data_index += c_) {
      *(im_inv++) = *(data_index + c_ - 1);
      *(im_inv++) = *(data_index + 1);
      *(im_inv++) = *data_index;
    }
  } else {
    LOG(FATAL) << "Unsupported format " << order_ << " to get inverse!";
  }
}

}  // namespace Shadow
