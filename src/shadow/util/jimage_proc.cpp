#include "shadow/util/jimage_proc.hpp"
#include "shadow/util/util.hpp"

namespace JImageProc {

void Filter1D(const JImage &im_src, JImage *im_filter, const float *kernel,
              int kernel_size, int direction) {
  if (im_src.data() == nullptr) Fatal("JImage src data is NULL!");
  if (im_filter == nullptr) Fatal("JImage filter is NULL!");

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  Order order = im_src.order();

  im_filter->Reshape(c_, h_, w_, order);

  const unsigned char *data_src = im_src.data();
  unsigned char *data_filter = im_filter->data();

  float val_c0, val_c1, val_c2, val_kernel;
  int p, p_loc, l_, im_index, center = kernel_size >> 1;

  for (int h = 0; h < h_; ++h) {
    for (int w = 0; w < w_; ++w) {
      val_c0 = 0.f, val_c1 = 0.f, val_c2 = 0.f;
      l_ = !direction ? w_ : h_;
      p = !direction ? w : h;
      for (int i = 0; i < kernel_size; ++i) {
        p_loc = std::abs(p - center + i);
        p_loc = p_loc < l_ ? p_loc : ((l_ << 1) - 1 - p_loc) % l_;
        im_index = !direction ? (w_ * h + p_loc) * c_ : (w_ * p_loc + w) * c_;
        val_kernel = kernel[i];
        val_c0 += data_src[im_index + 0] * val_kernel;
        if (order != kGray) {
          val_c1 += data_src[im_index + 1] * val_kernel;
          val_c2 += data_src[im_index + 2] * val_kernel;
        }
      }
      *data_filter++ =
          (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c0));
      if (order != kGray) {
        *data_filter++ =
            (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c1));
        *data_filter++ =
            (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c2));
      }
    }
  }
}

void Filter2D(const JImage &im_src, JImage *im_filter, const float *kernel,
              int height, int width) {
  if (im_src.data() == nullptr) Fatal("JImage src data is NULL!");
  if (im_filter == nullptr) Fatal("JImage filter is NULL!");

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  Order order = im_src.order();

  im_filter->Reshape(c_, h_, w_, order);

  const unsigned char *data_src = im_src.data();
  unsigned char *data_filter = im_filter->data();

  float val_c0, val_c1, val_c2, val_kernel;
  int im_h, im_w, im_index;

  for (int h = 0; h < h_; ++h) {
    for (int w = 0; w < w_; ++w) {
      val_c0 = 0.f, val_c1 = 0.f, val_c2 = 0.f;
      for (int k_h = 0; k_h < height; ++k_h) {
        for (int k_w = 0; k_w < width; ++k_w) {
          im_h = std::abs(h - (height >> 1) + k_h);
          im_w = std::abs(w - (width >> 1) + k_w);
          im_h = im_h < h_ ? im_h : ((h_ << 1) - 1 - im_h) % h_;
          im_w = im_w < w_ ? im_w : ((w_ << 1) - 1 - im_w) % w_;
          im_index = (w_ * im_h + im_w) * c_;
          val_kernel = kernel[k_h * width + k_w];
          val_c0 += data_src[im_index + 0] * val_kernel;
          if (order != kGray) {
            val_c1 += data_src[im_index + 1] * val_kernel;
            val_c2 += data_src[im_index + 2] * val_kernel;
          }
        }
      }
      *data_filter++ =
          (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c0));
      if (order != kGray) {
        *data_filter++ =
            (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c1));
        *data_filter++ =
            (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c2));
      }
    }
  }
}

void GaussianBlur(const JImage &im_src, JImage *im_blur, int kernel_size,
                  float sigma) {
  if (im_src.data() == nullptr) Fatal("JImage src data is NULL!");
  if (im_blur == nullptr) Fatal("JImage blur is NULL!");

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  Order order = im_src.order();

  im_blur->Reshape(c_, h_, w_, order);

  const unsigned char *data_src = im_src.data();
  unsigned char *data_blur = im_blur->data();

  float *kernel = new float[kernel_size];
  GetGaussianKernel(kernel, kernel_size, sigma);

  float val_c0, val_c1, val_c2, val_kernel;
  int im_h, im_w, im_index, center = kernel_size >> 1;

  float *data_w = new float[c_ * h_ * w_];
  float *data_w_index = data_w;
  for (int h = 0; h < h_; ++h) {
    for (int w = 0; w < w_; ++w) {
      val_c0 = 0.f, val_c1 = 0.f, val_c2 = 0.f;
      for (int k_w = 0; k_w < kernel_size; ++k_w) {
        im_w = std::abs(w - center + k_w);
        im_w = im_w < w_ ? im_w : ((w_ << 1) - 1 - im_w) % w_;
        im_index = (w_ * h + im_w) * c_;
        val_kernel = kernel[k_w];
        val_c0 += data_src[im_index + 0] * val_kernel;
        if (order != kGray) {
          val_c1 += data_src[im_index + 1] * val_kernel;
          val_c2 += data_src[im_index + 2] * val_kernel;
        }
      }
      *data_w_index++ = val_c0;
      if (order != kGray) {
        *data_w_index++ = val_c1;
        *data_w_index++ = val_c2;
      }
    }
  }

  for (int h = 0; h < h_; ++h) {
    for (int w = 0; w < w_; ++w) {
      val_c0 = 0.f, val_c1 = 0.f, val_c2 = 0.f;
      for (int k_h = 0; k_h < kernel_size; ++k_h) {
        im_h = std::abs(h - center + k_h);
        im_h = im_h < h_ ? im_h : ((h_ << 1) - 1 - im_h) % h_;
        im_index = (w_ * im_h + w) * c_;
        val_kernel = kernel[k_h];
        val_c0 += data_w[im_index + 0] * val_kernel;
        if (order != kGray) {
          val_c1 += data_w[im_index + 1] * val_kernel;
          val_c2 += data_w[im_index + 2] * val_kernel;
        }
      }
      *data_blur++ =
          (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c0));
      if (order != kGray) {
        *data_blur++ =
            (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c1));
        *data_blur++ =
            (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c2));
      }
    }
  }

  delete[] data_w;
  delete[] kernel;
}

void Canny(const JImage &im_src, JImage *im_canny, float thresh_low,
           float thresh_high, bool L2) {
  if (im_src.data() == nullptr) Fatal("JImage data is NULL!");

  im_src.CopyTo(im_canny);

  if (im_canny->order() != kGray) im_canny->Color2Gray();

  int h_ = im_canny->h_, w_ = im_canny->w_;
  unsigned char *data_ = im_canny->data();

  int *grad_x = new int[h_ * w_], *grad_y = new int[h_ * w_];
  int *magnitude = new int[h_ * w_];
  Gradient(*im_canny, grad_x, grad_y, magnitude, L2);

  if (L2) {
    if (thresh_low > 0) thresh_low *= thresh_low;
    if (thresh_high > 0) thresh_high *= thresh_high;
  }

//   0 - the pixel might belong to an edge
//   1 - the pixel can not belong to an edge
//   2 - the pixel does belong to an edge

#define CANNY_SHIFT 15
  const int TG22 = static_cast<int>(
      0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) + 0.5);

  memset(im_canny->data(), 0, sizeof(unsigned char) * h_ * w_);
  int index, val, g_x, g_y, x, y, prev_flag, tg22x, tg67x;
  for (int h = 1; h < h_ - 1; ++h) {
    prev_flag = 0;
    for (int w = 1; w < w_ - 1; ++w) {
      index = h * w_ + w;
      val = magnitude[index];
      if (val > thresh_low) {
        g_x = grad_x[index];
        g_y = grad_y[index];
        x = std::abs(g_x);
        y = std::abs(g_y) << CANNY_SHIFT;

        tg22x = x * TG22;
        if (y < tg22x) {
          if (val > magnitude[index - 1] && val >= magnitude[index + 1])
            goto __canny_set;
        } else {
          tg67x = tg22x + (x << (CANNY_SHIFT + 1));
          if (y > tg67x) {
            if (val > magnitude[index - w_] && val >= magnitude[index + w_])
              goto __canny_set;
          } else {
            int s = (g_x ^ g_y) < 0 ? -1 : 1;
            if (val > magnitude[index - w_ - s] &&
                val > magnitude[index + w_ + s])
              goto __canny_set;
          }
        }
      }
      prev_flag = 0;
      data_[index] = 1;
      continue;
    __canny_set:
      if (!prev_flag && val > thresh_high && data_[index - w_] != 2) {
        data_[index] = 2;
        prev_flag = 1;
      } else {
        data_[index] = 0;
      }
    }
  }

  for (int h = 1; h < h_ - 1; ++h) {
    for (int w = 1; w < w_ - 1; ++w) {
      index = h * w_ + w;
      if (data_[index] == 2) {
        if (!data_[index - w_ - 1]) data_[index - w_ - 1] = 2;
        if (!data_[index - w_]) data_[index - w_] = 2;
        if (!data_[index - w_ + 1]) data_[index - w_ + 1] = 2;
        if (!data_[index - 1]) data_[index - 1] = 2;
        if (!data_[index + 1]) data_[index + 1] = 2;
        if (!data_[index + w_ - 1]) data_[index + w_ - 1] = 2;
        if (!data_[index + w_]) data_[index + w_] = 2;
        if (!data_[index + w_ + 1]) data_[index + w_ + 1] = 2;
      }
    }
  }

  unsigned char *data_index = data_;
  for (int i = 0; i < h_ * w_; ++i, ++data_index) {
    *data_index = (unsigned char)-(*data_index >> 1);
  }

  delete[] grad_x;
  delete[] grad_y;
  delete[] magnitude;
}

void GetGaussianKernel(float *kernel, int n, float sigma) {
  const int SMALL_GAUSSIAN_SIZE = 7;
  static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] = {
      {1.f},
      {0.25f, 0.5f, 0.25f},
      {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
      {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}};

  const float *fixed_kernel =
      n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0
          ? small_gaussian_tab[n >> 1]
          : 0;

  float sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5f - 1) * 0.3f + 0.8f;
  float scale2X = -0.5f / (sigmaX * sigmaX);

  float sum = 0;
  for (int i = 0; i < n; ++i) {
    float x = i - (n - 1) * 0.5f;
    float t = fixed_kernel ? fixed_kernel[i] : std::exp(scale2X * x * x);
    kernel[i] = t;
    sum += kernel[i];
  }

  sum = 1.f / sum;
  for (int i = 0; i < n; ++i) {
    kernel[i] = kernel[i] * sum;
  }
}

void Gradient(const JImage &im_src, int *grad_x, int *grad_y, int *magnitude,
              bool L2) {
  if (im_src.data() == nullptr) Fatal("JImage data is NULL!");

  const unsigned char *data_src = im_src.data();
  int h_ = im_src.h_, w_ = im_src.w_;

  JImage *im_gray = nullptr;
  if (im_src.order() != kGray) {
    im_gray = new JImage();
    im_src.CopyTo(im_gray);
    data_src = im_gray->data();
  }

  memset(grad_x, 0, sizeof(int) * h_ * w_);
  memset(grad_y, 0, sizeof(int) * h_ * w_);
  memset(magnitude, 0, sizeof(int) * h_ * w_);

  int index;
  for (int h = 1; h < h_ - 1; ++h) {
    for (int w = 1; w < w_ - 1; ++w) {
      index = h * w_ + w;
      // grad_x[index] = data_src[index + 1] - data_src[index - 1];
      // grad_y[index] = data_src[index + w_] - data_src[index - w_];
      grad_x[index] = data_src[index + w_ + 1] + data_src[index - w_ + 1] +
                      (data_src[index + 1] << 1) - data_src[index + w_ - 1] -
                      data_src[index - w_ - 1] - (data_src[index - 1] << 1);
      grad_y[index] = data_src[index - w_ - 1] + data_src[index - w_ + 1] +
                      (data_src[index - w_] << 1) - data_src[index + w_ - 1] -
                      data_src[index + w_ + 1] - (data_src[index + w_] << 1);
      if (L2) {
        magnitude[index] = static_cast<int>(std::sqrt(
            grad_x[index] * grad_x[index] + grad_y[index] * grad_y[index]));
      } else {
        magnitude[index] = std::abs(grad_x[index]) + std::abs(grad_y[index]);
      }
    }
  }
  if (im_gray != nullptr) {
    im_gray->Release();
  }
}

}  // namespace JImageProc
