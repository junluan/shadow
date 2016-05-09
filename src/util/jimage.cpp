#include "jimage.hpp"
#include "util.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

struct Scalar {
  unsigned char r, g, b;
};

JImage::JImage() { data_ = nullptr; }

JImage::JImage(std::string im_path) {
  data_ = nullptr;
  Read(im_path);
}

JImage::JImage(int channel, int height, int width, Order order) {
  c_ = channel;
  h_ = height;
  w_ = width;
  order_ = order;
  data_ = new unsigned char[c_ * h_ * w_];
}

JImage::~JImage() { delete[] data_; }

void JImage::Read(std::string im_path) {
  if (data_ != nullptr)
    delete[] data_;
  data_ = stbi_load(im_path.c_str(), &w_, &h_, &c_, 3);
  if (data_ == nullptr)
    error("Failed to read image " + im_path);
  order_ = kRGB;
}

void JImage::Write(std::string im_path) {
  if (data_ == nullptr)
    error("JImage data is NULL!");
  int is_ok = -1;
  int step = w_ * c_;
  std::string path = change_extension(im_path, ".png");
  if (order_ == kRGB) {
    is_ok = stbi_write_png(path.c_str(), w_, h_, c_, data_, step);
  } else if (order_ == kBGR) {
    unsigned char *data_inv = new unsigned char[c_ * h_ * w_];
    GetInv(data_inv);
    is_ok = stbi_write_png(path.c_str(), w_, h_, c_, data_inv, step);
    delete[] data_inv;
  } else {
    error("Unsupported format to disk!");
  }
  if (!is_ok)
    error("Failed to write image to " + im_path);
}

void JImage::Show(std::string show_name) {
  if (data_ == nullptr)
    error("JImage data is NULL!");
#ifdef USE_OpenCV
  if (order_ == kRGB) {
    unsigned char *data_inv = new unsigned char[c_ * h_ * w_];
    GetInv(data_inv);
    cv::Mat im_mat(h_, w_, CV_8UC3, data_inv);
    cv::imshow(show_name, im_mat);
    cv::waitKey(0);
    delete[] data_inv;
  } else if (order_ == kBGR) {
    cv::Mat im_mat(h_, w_, CV_8UC3, data_);
    cv::imshow(show_name, im_mat);
    cv::waitKey(0);
  } else {
    error("Unsupported format to show!");
  }
#else
  warn("Not compiled with OpenCV, saving image to " + show_name + ".png");
  Write(show_name + ".png");
#endif
}

void JImage::CopyTo(JImage *im_copy) {
  if (data_ == nullptr)
    error("JImage data is NULL!");
  if (im_copy->data_ == nullptr) {
    im_copy->data_ = new unsigned char[c_ * h_ * w_];
  } else if (im_copy->h_ * im_copy->w_ < h_ * w_) {
    delete[] im_copy->data_;
    im_copy->data_ = new unsigned char[c_ * h_ * w_];
  }
  im_copy->c_ = c_;
  im_copy->h_ = h_;
  im_copy->w_ = w_;
  im_copy->order_ = order_;
  memcpy(im_copy->data_, data_, c_ * h_ * w_);
}

void JImage::Resize(JImage *im_res, int height, int width) {
  if (data_ == nullptr)
    error("JImage data is NULL!");
  if (im_res->data_ == nullptr) {
    im_res->data_ = new unsigned char[c_ * height * width];
  } else if (im_res->h_ * im_res->w_ < height * width) {
    delete[] im_res->data_;
    im_res->data_ = new unsigned char[c_ * height * width];
  }
  im_res->c_ = c_;
  im_res->h_ = height;
  im_res->w_ = width;
  im_res->order_ = order_;
  if (order_ != kRGB && order_ != kBGR)
    error("Unsupported format to resize!");
  stbir_resize_uint8(data_, w_, h_, w_ * c_, im_res->data_, im_res->w_,
                     im_res->h_, im_res->w_ * im_res->c_, c_);
}

void JImage::Crop(JImage *im_crop, Box crop) {
  if (data_ == nullptr)
    error("JImage data is NULL!");
  if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > 1 || crop.y + crop.h > 1)
    error("Crop region overflow!");
  int height = static_cast<int>(crop.h * h_);
  int width = static_cast<int>(crop.w * w_);
  if (im_crop->data_ == nullptr) {
    im_crop->data_ = new unsigned char[c_ * height * width];
  } else if (im_crop->h_ * im_crop->w_ < height * width) {
    delete[] im_crop->data_;
    im_crop->data_ = new unsigned char[c_ * height * width];
  }
  im_crop->c_ = c_;
  im_crop->h_ = height;
  im_crop->w_ = width;
  im_crop->order_ = order_;
  if (order_ != kRGB && order_ != kBGR)
    error("Unsupported format to crop!");
  int step_src = w_ * c_;
  int step_crop = im_crop->w_ * c_;
  int w_off = static_cast<int>(crop.x * w_);
  int h_off = static_cast<int>(crop.y * h_);
  unsigned char *index_src, *index_crop;
  for (int h = 0; h < im_crop->h_; ++h) {
    index_src = data_ + (h + h_off) * step_src + w_off * c_;
    index_crop = im_crop->data_ + h * step_crop;
    memcpy(index_crop, index_src, step_crop);
  }
}

void JImage::Filter2D(float *kernel, int height, int width) {
  unsigned char *data_f_ = new unsigned char[c_ * h_ * w_];
  for (int h = 0; h < h_; ++h) {
    for (int w = 0; w < w_; ++w) {
      float val_c0 = 0.f, val_c1 = 0.f, val_c2 = 0.f;
      int im_h, im_w, im_index, kernel_index;
      for (int k_h = 0; k_h < height; ++k_h) {
        for (int k_w = 0; k_w < width; ++k_w) {
          im_h = std::abs(h - height / 2 + k_h);
          im_w = std::abs(w - width / 2 + k_w);
          im_h = im_h < h_ ? im_h : 2 * h_ - 2 - im_h;
          im_w = im_w < w_ ? im_w : 2 * w_ - 2 - im_w;
          im_index = (w_ * im_h + im_w) * c_;
          kernel_index = k_h * width + k_w;
          val_c0 += data_[im_index + 0] * kernel[kernel_index];
          val_c1 += data_[im_index + 1] * kernel[kernel_index];
          val_c2 += data_[im_index + 2] * kernel[kernel_index];
        }
      }
      int offset = (w_ * h + w) * c_;
      data_f_[offset + 0] =
          (unsigned char)constrain(0, 255, static_cast<int>(val_c0));
      data_f_[offset + 1] =
          (unsigned char)constrain(0, 255, static_cast<int>(val_c1));
      data_f_[offset + 2] =
          (unsigned char)constrain(0, 255, static_cast<int>(val_c2));
    }
  }
  memcpy(data_, data_f_, c_ * h_ * w_);
}

void JImage::FromI420(unsigned char *src_y, unsigned char *src_u,
                      unsigned char *src_v, int src_h, int src_w,
                      int src_stride) {
  if (data_ == nullptr) {
    data_ = new unsigned char[3 * src_h * src_w];
  } else if (h_ * w_ < src_h * src_w) {
    delete[] data_;
    data_ = new unsigned char[3 * src_h * src_w];
  }
  c_ = 3;
  h_ = src_h;
  w_ = src_w;
  order_ = kRGB;
  for (int h = 0; h < h_; ++h) {
    for (int w = 0; w < w_; ++w) {
      int y = src_y[h * src_stride + w];
      int u = src_u[(h >> 1) * (src_stride >> 1) + (w >> 1)];
      int v = src_v[(h >> 1) * (src_stride >> 1) + (w >> 1)];
      u -= 128;
      v -= 128;
      int r = y + v + ((v * 103) >> 8);
      int g = y - ((u * 88) >> 8) + ((v * 183) >> 8);
      int b = y + u + ((u * 198) >> 8);

      int offset = (w_ * h + w) * c_;
      data_[offset + 0] = (unsigned char)constrain(0, 255, r);
      data_[offset + 1] = (unsigned char)constrain(0, 255, g);
      data_[offset + 2] = (unsigned char)constrain(0, 255, b);
    }
  }
}

void JImage::FromI420WithCropResize(unsigned char *src_y, unsigned char *src_u,
                                    unsigned char *src_v, int src_h, int src_w,
                                    int src_stride, Box roi, int resize_h,
                                    int resize_w, float *batch_data) {
  Box roi_p;
  roi_p.x = roi.x * src_w;
  roi_p.y = roi.y * src_h;
  roi_p.w = roi.w * src_w;
  roi_p.h = roi.h * src_h;

  float step_w = roi_p.w / resize_w;
  float step_h = roi_p.h / resize_h;
  int step_ch = resize_h * resize_w;

  for (int h = 0; h < resize_h; ++h) {
    for (int w = 0; w < resize_w; ++w) {
      int s_h = static_cast<int>(roi_p.y + step_h * h);
      int s_w = static_cast<int>(roi_p.x + step_w * w);

      int y = src_y[s_h * src_stride + s_w];
      int u = src_u[(s_h >> 1) * (src_stride >> 1) + (s_w >> 1)];
      int v = src_v[(s_h >> 1) * (src_stride >> 1) + (s_w >> 1)];
      u -= 128;
      v -= 128;
      int r = y + v + ((v * 103) >> 8);
      int g = y - ((u * 88) >> 8) + ((v * 183) >> 8);
      int b = y + u + ((u * 198) >> 8);

      int offset = h * resize_w + w;
      batch_data[offset + step_ch * 0] = (unsigned char)constrain(0, 255, r);
      batch_data[offset + step_ch * 1] = (unsigned char)constrain(0, 255, g);
      batch_data[offset + step_ch * 2] = (unsigned char)constrain(0, 255, b);
    }
  }
}

#ifdef USE_OpenCV
void JImage::FromMat(const cv::Mat &im_mat) {
  if (data_ == nullptr) {
    data_ = new unsigned char[im_mat.channels() * im_mat.rows * im_mat.cols];
  } else if (h_ * w_ < im_mat.rows * im_mat.cols) {
    delete[] data_;
    data_ = new unsigned char[im_mat.channels() * im_mat.rows * im_mat.cols];
  }
  c_ = im_mat.channels();
  h_ = im_mat.rows;
  w_ = im_mat.cols;
  order_ = kBGR;
  memcpy(data_, im_mat.data, c_ * h_ * w_);
}

void JImage::FromMatWithCropResize(const cv::Mat &im_mat, Box roi, int resize_h,
                                   int resize_w, float *batch_data) {
  Box roi_p;
  roi_p.x = roi.x * im_mat.cols;
  roi_p.y = roi.y * im_mat.rows;
  roi_p.w = roi.w * im_mat.cols;
  roi_p.h = roi.h * im_mat.rows;

  float step_w = roi_p.w / resize_w;
  float step_h = roi_p.h / resize_h;
  int step_ch = resize_h * resize_w;
  int step = im_mat.cols * im_mat.channels();

  for (int h = 0; h < resize_h; ++h) {
    for (int w = 0; w < resize_w; ++w) {
      int s_h = static_cast<int>(roi_p.y + step_h * h);
      int s_w = static_cast<int>(roi_p.x + step_w * w);
      int src_offset = s_h * step + s_w * im_mat.channels();
      int offset = h * resize_w + w;
      batch_data[offset + step_ch * 0] = im_mat.data[src_offset + 2];
      batch_data[offset + step_ch * 1] = im_mat.data[src_offset + 1];
      batch_data[offset + step_ch * 2] = im_mat.data[src_offset + 0];
    }
  }
}
#endif

void JImage::Rectangle(const VecBox &boxes, bool console_show) {
  for (int b = 0; b < boxes.size(); ++b) {
    if (boxes[b].class_index == -1)
      continue;

    Box box = boxes[b];
    int x1 = constrain(0, w_ - 1, static_cast<int>(box.x));
    int y1 = constrain(0, h_ - 1, static_cast<int>(box.y));
    int x2 = constrain(x1, w_ - 1, static_cast<int>(x1 + box.w));
    int y2 = constrain(y1, h_ - 1, static_cast<int>(y1 + box.h));

    Scalar scalar;
    if (box.class_index == 0)
      scalar = {0, 255, 0};
    else
      scalar = {0, 0, 255};

    for (int i = x1; i <= x2; ++i) {
      int offset = (w_ * y1 + i) * c_;
      data_[offset + 0] = scalar.r;
      data_[offset + 1] = scalar.g;
      data_[offset + 2] = scalar.b;
      offset = (w_ * y2 + i) * c_;
      data_[offset + 0] = scalar.r;
      data_[offset + 1] = scalar.g;
      data_[offset + 2] = scalar.b;
    }
    for (int i = y1; i <= y2; ++i) {
      int offset = (w_ * i + x1) * c_;
      data_[offset + 0] = scalar.r;
      data_[offset + 1] = scalar.g;
      data_[offset + 2] = scalar.b;
      offset = (w_ * i + x2) * c_;
      data_[offset + 0] = scalar.r;
      data_[offset + 1] = scalar.g;
      data_[offset + 2] = scalar.b;
    }
    if (console_show) {
      std::cout << "x = " << box.x << ", y = " << box.y << ", w = " << box.w
                << ", h = " << box.h << ", score = " << box.score
                << ", label = " << box.class_index << std::endl;
    }
  }
}

void JImage::GetBatchData(float *batch_data) {
  if (data_ == nullptr)
    error("JImage data is NULL!");
  bool is_rgb = false;
  if (order_ == kRGB) {
    is_rgb = true;
  } else if (order_ == kBGR) {
    is_rgb = false;
  } else {
    error("Unsupported format to get batch data!");
  }
  int ch_src, offset, count = 0, step = w_ * c_;
  for (int c = 0; c < c_; ++c) {
    for (int h = 0; h < h_; ++h) {
      for (int w = 0; w < w_; ++w) {
        ch_src = is_rgb ? c : c_ - c - 1;
        offset = h * step + w * c_;
        batch_data[count++] = data_[offset + ch_src];
      }
    }
  }
}

void JImage::Release() { delete[] data_; }

void JImage::GetInv(unsigned char *im_inv) {
  bool is_rgb2bgr = false;
  if (order_ == kRGB) {
    is_rgb2bgr = true;
  } else if (order_ == kBGR) {
    is_rgb2bgr = false;
  } else {
    error("Unsupported format to inverse!");
  }
  int ch_src, ch_inv, offset, step = w_ * c_;
  for (int c = 0; c < c_; ++c) {
    for (int h = 0; h < h_; ++h) {
      for (int w = 0; w < w_; ++w) {
        ch_src = is_rgb2bgr ? c : c_ - c - 1;
        ch_inv = c_ - 1 - ch_src;
        offset = h * step + w * c_;
        im_inv[offset + ch_inv] = data_[offset + ch_src];
      }
    }
  }
}
