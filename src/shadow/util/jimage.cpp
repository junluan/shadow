#include "shadow/util/jimage.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

void JImage::Read(const std::string im_path) {
  if (data_ != nullptr) delete[] data_;
  data_ = stbi_load(im_path.c_str(), &w_, &h_, &c_, 3);
  if (data_ == nullptr) Fatal("Failed to read image " + im_path);
  order_ = kRGB;
}

void JImage::Write(const std::string im_path) {
  if (data_ == nullptr) Fatal("JImage data is NULL!");
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
    Fatal("Unsupported format to disk!");
  }
  if (!is_ok) Fatal("Failed to write image to " + im_path);
}

void JImage::Show(const std::string show_name, int wait_time) {
  if (data_ == nullptr) Fatal("JImage data is NULL!");
#if defined(USE_OpenCV)
  cv::namedWindow(show_name, cv::WINDOW_NORMAL);
  if (order_ == kRGB) {
    unsigned char *data_inv = new unsigned char[c_ * h_ * w_];
    GetInv(data_inv);
    cv::Mat im_mat(h_, w_, CV_8UC3, data_inv);
    cv::imshow(show_name, im_mat);
    delete[] data_inv;
  } else if (order_ == kBGR) {
    cv::Mat im_mat(h_, w_, CV_8UC3, data_);
    cv::imshow(show_name, im_mat);
  } else {
    Fatal("Unsupported format to show!");
  }
  cv::waitKey(wait_time);
#else
  Warning("Not compiled with OpenCV, saving image to " + show_name + ".png");
  Write(show_name + ".png");
#endif
}

void JImage::CopyTo(JImage *im_copy) {
  if (data_ == nullptr) Fatal("JImage data is NULL!");
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
  if (data_ == nullptr) Fatal("JImage data is NULL!");
  if (order_ != kRGB && order_ != kBGR) Fatal("Unsupported format to resize!");
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
  float step_h = static_cast<float>(h_) / im_res->h_;
  float step_w = static_cast<float>(w_) / im_res->w_;
  int src_step = w_ * c_, dst_step = im_res->w_ * c_;
  for (int c = 0; c < c_; ++c) {
    for (int h = 0; h < im_res->h_; ++h) {
      for (int w = 0; w < im_res->w_; ++w) {
        int s_h = static_cast<int>(step_h * h);
        int s_w = static_cast<int>(step_w * w);
        int src_offset = s_h * src_step + s_w * c_;
        int dst_offset = h * dst_step + w * c_;
        im_res->data_[dst_offset + c] = data_[src_offset + c];
      }
    }
  }
}

void JImage::Crop(JImage *im_crop, RectF crop) {
  if (data_ == nullptr) Fatal("JImage data is NULL!");
  if (order_ != kRGB && order_ != kBGR) Fatal("Unsupported format to crop!");
  if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > 1 || crop.y + crop.h > 1)
    Fatal("Crop region overflow!");
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

void JImage::CropWithResize(JImage *im_res, RectF crop, int height, int width) {
  if (data_ == nullptr) Fatal("JImage data is NULL!");
  if (order_ != kRGB && order_ != kBGR)
    Fatal("Unsupported format to crop and resize!");
  if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > 1 || crop.y + crop.h > 1)
    Fatal("Crop region overflow!");
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
  float step_h = crop.h * h_ / height;
  float step_w = crop.w * w_ / width;
  float h_off = crop.y * h_, w_off = crop.x * w_;
  int src_step = w_ * c_, dst_step = width * c_;
  for (int c = 0; c < c_; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int s_h = static_cast<int>(h_off + step_h * h);
        int s_w = static_cast<int>(w_off + step_w * w);
        int src_offset = s_h * src_step + s_w * c_;
        int dst_offset = h * dst_step + w * c_;
        im_res->data_[dst_offset + c] = data_[src_offset + c];
      }
    }
  }
}

void JImage::Filter2D(const float *kernel, int height, int width) {
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

#if defined(USE_OpenCV)
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

void JImage::FromMatWithCropResize(const cv::Mat &im_mat, const RectF crop,
                                   int resize_h, int resize_w,
                                   float *batch_data) {
  RectF crop_p;
  crop_p.x = crop.x * im_mat.cols;
  crop_p.y = crop.y * im_mat.rows;
  crop_p.w = crop.w * im_mat.cols;
  crop_p.h = crop.h * im_mat.rows;

  float step_w = crop_p.w / resize_w;
  float step_h = crop_p.h / resize_h;
  int step_ch = resize_h * resize_w;
  int ch_offset, ch_src, src_step = im_mat.cols * im_mat.channels();
  for (int c = 0; c < 3; ++c) {
    ch_offset = step_ch * c;
    ch_src = c_ - c - 1;
    for (int h = 0; h < resize_h; ++h) {
      for (int w = 0; w < resize_w; ++w) {
        int s_h = static_cast<int>(crop_p.y + step_h * h);
        int s_w = static_cast<int>(crop_p.x + step_w * w);
        int src_offset = s_h * src_step + s_w * im_mat.channels();
        int offset = h * resize_w + w;
        batch_data[offset + ch_offset] = im_mat.data[src_offset + ch_src];
      }
    }
  }
}
#endif

#define CLIP(x) (unsigned char)((x) & (~255) ? ((-x) >> 31) : (x))
#define fix(x, n) static_cast<int>((x) * (1 << (n)) + 0.5)
#define yuvYr fix(0.299, 10)
#define yuvYg fix(0.587, 10)
#define yuvYb fix(0.114, 10)
#define yuvCr fix(0.713, 10)
#define yuvCb fix(0.564, 10)

void RGB2I420(unsigned char *src_bgr, int src_h, int src_w, int src_step,
              Order order, unsigned char *dst_i420) {
  int loc_r = 0, loc_g = 1, loc_b = 2;
  if (order == kRGB) {
    loc_r = 0;
    loc_g = 1;
    loc_b = 2;
  } else if (order == kBGR) {
    loc_r = 2;
    loc_g = 1;
    loc_b = 0;
  } else {
    Fatal("Unsupported order to convert to I420!");
  }
  int r, g, b;
  int dst_h = src_h / 2, dst_w = src_w / 2;
  int uv_offset = src_h * src_w, uv_step = dst_h * dst_w;
  for (int h = 0; h < dst_h; ++h) {
    for (int w = 0; w < dst_w; ++w) {
      int s_h = 2 * h, s_w = 2 * w, y, cb = 0, cr = 0;
      for (int h_off = 0; h_off < 2; ++h_off) {
        for (int w_off = 0; w_off < 2; ++w_off) {
          int src_offset = (s_h + h_off) * src_step + (s_w + w_off) * 3;
          int y_offset = (s_h + h_off) * src_w + s_w + w_off;
          r = src_bgr[src_offset + loc_r];
          g = src_bgr[src_offset + loc_g];
          b = src_bgr[src_offset + loc_b];
          y = (b * yuvYb + g * yuvYg + r * yuvYr) >> 10;
          cb += ((b - y) * yuvCb + (128 << 10)) >> 10;
          cr += ((r - y) * yuvCr + (128 << 10)) >> 10;
          dst_i420[y_offset] = (unsigned char)y;
        }
      }
      cb = CLIP((cb >> 2));
      cr = CLIP((cr >> 2));
      int offset = uv_offset + h * dst_w + w;
      dst_i420[offset] = (unsigned char)cb;
      dst_i420[offset + uv_step] = (unsigned char)cr;
    }
  }
}

void I4202RGB(unsigned char *src_y, unsigned char *src_u, unsigned char *src_v,
              int src_h, int src_w, int src_step, unsigned char *dst_rgb,
              Order order) {
  for (int h = 0; h < src_h; ++h) {
    for (int w = 0; w < src_w; ++w) {
      int y = src_y[h * src_step + w];
      int u = src_u[(h >> 1) * (src_step >> 1) + (w >> 1)];
      int v = src_v[(h >> 1) * (src_step >> 1) + (w >> 1)];
      u -= 128;
      v -= 128;
      int r = y + v + ((v * 103) >> 8);
      int g = y - ((u * 88) >> 8) + ((v * 183) >> 8);
      int b = y + u + ((u * 198) >> 8);

      int offset = (src_w * h + w) * 3;
      dst_rgb[offset + 1] = (unsigned char)constrain(0, 255, g);
      if (order == kRGB) {
        dst_rgb[offset + 0] = (unsigned char)constrain(0, 255, r);
        dst_rgb[offset + 2] = (unsigned char)constrain(0, 255, b);
      } else if (order == kBGR) {
        dst_rgb[offset + 0] = (unsigned char)constrain(0, 255, b);
        dst_rgb[offset + 2] = (unsigned char)constrain(0, 255, r);
      } else {
        Fatal("Unsupported format to convert i420 to rgb!");
      }
    }
  }
}

void JImage::Rectangle(const Box &box, const Scalar scalar, bool console_show) {
  int x1 = constrain(0, w_ - 1, static_cast<int>(box.x));
  int y1 = constrain(0, h_ - 1, static_cast<int>(box.y));
  int x2 = constrain(x1, w_ - 1, static_cast<int>(x1 + box.w));
  int y2 = constrain(y1, h_ - 1, static_cast<int>(y1 + box.h));

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

void JImage::Rectangle(const VecBox &boxes, const Scalar scalar,
                       bool console_show) {
  for (int b = 0; b < boxes.size(); ++b) {
    Rectangle(boxes[b], scalar, console_show);
  }
}

void JImage::GetBatchData(float *batch_data) {
  if (data_ == nullptr) Fatal("JImage data is NULL!");
  bool is_rgb = false;
  if (order_ == kRGB) {
    is_rgb = true;
  } else if (order_ == kBGR) {
    is_rgb = false;
  } else {
    Fatal("Unsupported format to get batch data!");
  }
  int ch_src, offset, count = 0, step = w_ * c_;
  for (int c = 0; c < c_; ++c) {
    ch_src = is_rgb ? c : c_ - c - 1;
    for (int h = 0; h < h_; ++h) {
      for (int w = 0; w < w_; ++w) {
        offset = h * step + w * c_;
        batch_data[count++] = data_[offset + ch_src];
      }
    }
  }
}

void JImage::Release() {
  if (data_ != nullptr) delete[] data_;
}

void JImage::GetInv(unsigned char *im_inv) {
  bool is_rgb2bgr = false;
  if (order_ == kRGB) {
    is_rgb2bgr = true;
  } else if (order_ == kBGR) {
    is_rgb2bgr = false;
  } else {
    Fatal("Unsupported format to inverse!");
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
