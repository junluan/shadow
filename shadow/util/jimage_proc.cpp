#include "jimage_proc.hpp"
#include "util.hpp"

namespace Shadow {

namespace JImageProc {

VecPointI GetLinePoints(const PointI &start, const PointI &end, int step,
                        int slice_axis) {
  PointI start_p(start), end_p(end);

  float delta_x = end_p.x - start_p.x, delta_y = end_p.y - start_p.y;

  if (std::abs(delta_x) < 1 && std::abs(delta_y) < 1) {
    return VecPointI(1, start_p);
  }

  bool steep = false;
  if (slice_axis == 1 ||
      (slice_axis == -1 && std::abs(delta_y) > std::abs(delta_x))) {
    steep = true;
    int t = start_p.x;
    start_p.x = start_p.y;
    start_p.y = t;
    t = end_p.x;
    end_p.x = end_p.y;
    end_p.y = t;
  }

  if (start_p.x > end_p.x) {
    PointI t(start_p);
    start_p = end_p;
    end_p = t;
  }

  delta_x = end_p.x - start_p.x, delta_y = end_p.y - start_p.y;

  float step_y = delta_y / delta_x;
  VecPointI points;
  for (int x = start_p.x; x <= end_p.x; x += step) {
    int y = Util::round(start_p.y + (x - start_p.x) * step_y);
    if (steep) {
      points.emplace_back(y, x);
    } else {
      points.emplace_back(x, y);
    }
  }

  return points;
}

template <typename T>
void Line(JImage *im, const Point<T> &start, const Point<T> &end,
          const Scalar &scalar) {
  CHECK_NOTNULL(im);
  CHECK_NOTNULL(im->data());

  int c_ = im->c_, h_ = im->h_, w_ = im->w_;
  const auto &order_ = im->order();

  if (order_ != kGray && order_ != kRGB && order_ != kBGR) {
    LOG(FATAL) << "Unsupported format " << order_ << " to draw line!";
  }

  auto *data = im->data();

  int loc_r = 0, loc_g = 1, loc_b = 2;
  if (order_ == kRGB) {
    loc_r = 0;
    loc_g = 1;
    loc_b = 2;
  } else if (order_ == kBGR) {
    loc_r = 2;
    loc_g = 1;
    loc_b = 0;
  }

  auto gray = std::max(std::max(scalar.r, scalar.g), scalar.b);

  const auto &points = GetLinePoints(PointI(start), PointI(end));

  int offset, x, y;
  for (const auto &point : points) {
    x = point.x, y = point.y;
    if (x < 0 || y < 0 || x >= w_ || y >= h_) continue;
    offset = (w_ * y + x) * c_;
    if (order_ == kGray) {
      data[offset] = gray;
    } else {
      data[offset + loc_r] = scalar.r;
      data[offset + loc_g] = scalar.g;
      data[offset + loc_b] = scalar.b;
    }
  }
}

template <typename T>
void Rectangle(JImage *im, const Rect<T> &rect, const Scalar &scalar) {
  CHECK_NOTNULL(im);
  CHECK_NOTNULL(im->data());

  int h_ = im->h_, w_ = im->w_;
  const auto &order_ = im->order();

  if (order_ != kGray && order_ != kRGB && order_ != kBGR) {
    LOG(FATAL) << "Unsupported format " << order_ << " to draw rectangle!";
  }

  RectI rectI(rect);

  int x1 = Util::constrain(0, w_ - 1, rectI.x);
  int y1 = Util::constrain(0, h_ - 1, rectI.y);
  int x2 = Util::constrain(x1, w_ - 1, x1 + rectI.w);
  int y2 = Util::constrain(y1, h_ - 1, y1 + rectI.h);

  Line(im, PointI(x1, y1), PointI(x2, y1), scalar);
  Line(im, PointI(x1, y1), PointI(x1, y2), scalar);
  Line(im, PointI(x1, y2), PointI(x2, y2), scalar);
  Line(im, PointI(x2, y1), PointI(x2, y2), scalar);
}

void Color2Gray(const JImage &im_src, JImage *im_gray,
                const Transformer &transformer) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_gray);
  CHECK_NE(im_src.data(), im_gray->data());

  int h_ = im_src.h_, w_ = im_src.w_, spatial_dim_ = h_ * w_;
  const auto &order_ = im_src.order();

  im_gray->Reshape(1, h_, w_, kGray);

  const auto *data_src = im_src.data();
  auto *data_gray = im_gray->data();

  if (transformer == kRGB2Gray) {
    CHECK((order_ == kRGB));
    int index = 0;
    for (int i = 0; i < spatial_dim_; ++i, index += 3) {
      auto r = data_src[index + 0];
      auto g = data_src[index + 1];
      auto b = data_src[index + 2];
      auto val = static_cast<int>(0.299f * r + 0.587f * g + 0.114f * b);
      *data_gray++ = (unsigned char)Util::constrain(0, 255, val);
    }
  } else if (transformer == kBGR2Gray) {
    CHECK((order_ == kBGR));
    int index = 0;
    for (int i = 0; i < spatial_dim_; ++i, index += 3) {
      auto r = data_src[index + 2];
      auto g = data_src[index + 1];
      auto b = data_src[index + 0];
      auto val = static_cast<int>(0.299f * r + 0.587f * g + 0.114f * b);
      *data_gray++ = (unsigned char)Util::constrain(0, 255, val);
    }
  } else if (transformer == kI4202Gray) {
    CHECK((order_ == kI420));
    memcpy(data_gray, data_src, h_ * w_ * sizeof(unsigned char));
  } else {
    LOG(FATAL) << "Unsupported source format " << order_
               << ", currently supported: kRGB2Gray, kBGR2Gray, kI4202Gray";
  }
}

void RGB2BGR(const JImage &im_src, JImage *im_dst,
             const Transformer &transformer) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_dst);

  int h_ = im_src.h_, w_ = im_src.w_, spatial_dim_ = h_ * w_;
  const auto &order_ = im_src.order();

  if (transformer == kRGB2BGR) {
    CHECK((order_ == kRGB));
    im_dst->Reshape(3, h_, w_, kBGR);
  } else if (transformer == kBGR2RGB) {
    CHECK((order_ == kBGR));
    im_dst->Reshape(3, h_, w_, kRGB);
  } else {
    LOG(FATAL) << "Unsupported source format " << order_
               << ", currently supported: kRGB2BGR, kBGR2RGB";
  }

  const auto *data_src = im_src.data();
  auto *data_dst = im_dst->data();

  for (int i = 0; i < spatial_dim_; ++i, data_src += 3, data_dst += 3) {
    auto c_0 = *data_src, c_1 = *(data_src + 1), c_2 = *(data_src + 2);
    *(data_dst + 0) = c_2;
    *(data_dst + 1) = c_1;
    *(data_dst + 2) = c_0;
  }
}

#define CLIP(x) static_cast<unsigned char>((x) & (~255) ? ((-(x)) >> 31) : (x))
#define fix(x, n) static_cast<int>((x) * (1 << (n)) + 0.5)
#define yuvYr fix(0.299, 10)
#define yuvYg fix(0.587, 10)
#define yuvYb fix(0.114, 10)
#define yuvCr fix(0.713, 10)
#define yuvCb fix(0.564, 10)

void RGB2I420(const JImage &im_src, JImage *im_i420,
              const Transformer &transformer) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_i420);
  CHECK_NE(im_src.data(), im_i420->data());

  int src_h_ = (im_src.h_ >> 1) << 1, src_w_ = (im_src.w_ >> 1) << 1;
  int src_step_ = im_src.w_ * 3;
  const auto &order_ = im_src.order();

  int loc_r = 0, loc_g = 1, loc_b = 2;
  if (transformer == kRGB2I420) {
    CHECK((order_ == kRGB));
    loc_r = 0, loc_g = 1, loc_b = 2;
  } else if (transformer == kBGR2I420) {
    CHECK((order_ == kBGR));
    loc_r = 2, loc_g = 1, loc_b = 0;
  } else {
    LOG(FATAL) << "Unsupported source format " << order_
               << ", currently supported: kRGB2I420, kBGR2I420";
  }

  im_i420->Reshape(3, src_h_, src_w_, kI420);

  const auto *data_src = im_src.data();
  auto *data_i420 = im_i420->data();

  int r, g, b;
  int dst_h = src_h_ >> 1, dst_w = src_w_ >> 1;
  int uv_offset = src_h_ * src_w_, uv_step = dst_h * dst_w;
  int s_h, s_w, y, h_off, w_off, src_offset, y_offset;
  int cb, cb_0, cb_1, cb_2, cb_3, cr, cr_0, cr_1, cr_2, cr_3;
  for (int h = 0; h < dst_h; ++h) {
    for (int w = 0; w < dst_w; ++w) {
      s_h = h << 1, s_w = w << 1;

      h_off = 0, w_off = 0;
      src_offset = (s_h + h_off) * src_step_ + (s_w + w_off) * 3;
      y_offset = (s_h + h_off) * src_w_ + s_w + w_off;
      r = data_src[src_offset + loc_r];
      g = data_src[src_offset + loc_g];
      b = data_src[src_offset + loc_b];
      y = (b * yuvYb + g * yuvYg + r * yuvYr) >> 10;
      cb_0 = ((b - y) * yuvCb + (128 << 10)) >> 10;
      cr_0 = ((r - y) * yuvCr + (128 << 10)) >> 10;
      data_i420[y_offset] = (unsigned char)y;

      h_off = 0, w_off = 1;
      src_offset = (s_h + h_off) * src_step_ + (s_w + w_off) * 3;
      y_offset = (s_h + h_off) * src_w_ + s_w + w_off;
      r = data_src[src_offset + loc_r];
      g = data_src[src_offset + loc_g];
      b = data_src[src_offset + loc_b];
      y = (b * yuvYb + g * yuvYg + r * yuvYr) >> 10;
      cb_1 = ((b - y) * yuvCb + (128 << 10)) >> 10;
      cr_1 = ((r - y) * yuvCr + (128 << 10)) >> 10;
      data_i420[y_offset] = (unsigned char)y;

      h_off = 1, w_off = 0;
      src_offset = (s_h + h_off) * src_step_ + (s_w + w_off) * 3;
      y_offset = (s_h + h_off) * src_w_ + s_w + w_off;
      r = data_src[src_offset + loc_r];
      g = data_src[src_offset + loc_g];
      b = data_src[src_offset + loc_b];
      y = (b * yuvYb + g * yuvYg + r * yuvYr) >> 10;
      cb_2 = ((b - y) * yuvCb + (128 << 10)) >> 10;
      cr_2 = ((r - y) * yuvCr + (128 << 10)) >> 10;
      data_i420[y_offset] = (unsigned char)y;

      h_off = 1, w_off = 1;
      src_offset = (s_h + h_off) * src_step_ + (s_w + w_off) * 3;
      y_offset = (s_h + h_off) * src_w_ + s_w + w_off;
      r = data_src[src_offset + loc_r];
      g = data_src[src_offset + loc_g];
      b = data_src[src_offset + loc_b];
      y = (b * yuvYb + g * yuvYg + r * yuvYr) >> 10;
      cb_3 = ((b - y) * yuvCb + (128 << 10)) >> 10;
      cr_3 = ((r - y) * yuvCr + (128 << 10)) >> 10;
      data_i420[y_offset] = (unsigned char)y;

      cb = CLIP(((cb_0 + cb_1 + cb_2 + cb_3) >> 2));
      cr = CLIP(((cr_0 + cr_1 + cr_2 + cr_3) >> 2));
      int offset = uv_offset + h * dst_w + w;
      data_i420[offset] = (unsigned char)cb;
      data_i420[offset + uv_step] = (unsigned char)cr;
    }
  }
}

void I4202RGB(const JImage &im_src, JImage *im_dst,
              const Transformer &transformer) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_dst);
  CHECK_NE(im_src.data(), im_dst->data());

  int src_h_ = im_src.h_, src_w_ = im_src.w_, spatial_dim_ = src_h_ * src_w_;
  const auto &order_ = im_src.order();

  CHECK((order_ == kI420));
  int loc_r = 0, loc_g = 1, loc_b = 2;
  if (transformer == kI4202RGB) {
    loc_r = 0, loc_g = 1, loc_b = 2;
    im_dst->Reshape(3, src_h_, src_w_, kRGB);
  } else if (transformer == kI4202BGR) {
    loc_r = 2, loc_g = 1, loc_b = 0;
    im_dst->Reshape(3, src_h_, src_w_, kBGR);
  } else {
    LOG(FATAL) << "Unsupported transformer " << transformer
               << ", currently supported: kI4202RGB, kI4202BGR";
  }

  const auto *data_src_y = im_src.data();
  const auto *data_src_u = data_src_y + spatial_dim_;
  const auto *data_src_v = data_src_y + spatial_dim_ * 5 / 4;
  auto *data_dst = im_dst->data();

  for (int h = 0; h < src_h_; ++h) {
    for (int w = 0; w < src_w_; ++w) {
      int index_y = h * im_src.w_ + w;
      int index_uv = (h >> 1) * (im_src.w_ >> 1) + (w >> 1);
      int y = data_src_y[index_y];
      int u = data_src_u[index_uv];
      int v = data_src_v[index_uv];
      u -= 128, v -= 128;
      int r = y + v + ((v * 103) >> 8);
      int g = y - ((u * 88) >> 8) - ((v * 183) >> 8);
      int b = y + u + ((u * 198) >> 8);

      int index = (src_w_ * h + w) * 3;
      data_dst[index + loc_r] = (unsigned char)Util::constrain(0, 255, r);
      data_dst[index + loc_g] = (unsigned char)Util::constrain(0, 255, g);
      data_dst[index + loc_b] = (unsigned char)Util::constrain(0, 255, b);
    }
  }
}

// Format transform
void FormatTransform(const JImage &im_src, JImage *im_dst,
                     const Transformer &transformer) {
  switch (transformer) {
    case kRGB2Gray:
    case kBGR2Gray:
    case kI4202Gray: {
      Color2Gray(im_src, im_dst, transformer);
      break;
    }
    case kRGB2BGR:
    case kBGR2RGB: {
      RGB2BGR(im_src, im_dst, transformer);
      break;
    }
    case kRGB2I420:
    case kBGR2I420: {
      RGB2I420(im_src, im_dst, transformer);
      break;
    }
    case kI4202RGB:
    case kI4202BGR: {
      I4202RGB(im_src, im_dst, transformer);
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported transformer: " << transformer;
      break;
    }
  }
}

// Resize and Crop.
void Resize(const JImage &im_src, JImage *im_res, int height, int width) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_res);
  CHECK_NE(im_src.data(), im_res->data());

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  const auto &order_ = im_src.order();

  if (order_ != kGray && order_ != kRGB && order_ != kBGR) {
    LOG(FATAL) << "Unsupported format " << order_ << " to resize!";
  }

  im_res->Reshape(c_, height, width, order_);

  const auto *data_src = im_src.data();
  auto *data_gray = im_res->data();

  float step_h = static_cast<float>(h_) / im_res->h_;
  float step_w = static_cast<float>(w_) / im_res->w_;
  int s_h, s_w, src_offset, dst_offset;
  int src_step = w_ * c_, dst_step = im_res->w_ * c_;
  for (int c = 0; c < c_; ++c) {
    for (int h = 0; h < im_res->h_; ++h) {
      for (int w = 0; w < im_res->w_; ++w) {
        s_h = static_cast<int>(step_h * h);
        s_w = static_cast<int>(step_w * w);
        src_offset = s_h * src_step + s_w * c_;
        dst_offset = h * dst_step + w * c_;
        data_gray[dst_offset + c] = data_src[src_offset + c];
      }
    }
  }
}

template <typename T>
void Crop(const JImage &im_src, JImage *im_crop, const Rect<T> &crop) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_crop);
  CHECK_NE(im_src.data(), im_crop->data());

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  const auto &order_ = im_src.order();

  if (order_ != kGray && order_ != kRGB && order_ != kBGR) {
    LOG(FATAL) << "Unsupported format " << order_ << " to crop!";
  }

  if (crop.w <= 1 && crop.h <= 1) {
    if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > 1 ||
        crop.y + crop.h > 1) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else if (crop.w > 1 && crop.h > 1) {
    if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > w_ ||
        crop.y + crop.h > h_) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else {
    LOG(FATAL) << "Crop scale must be the same!";
  }

  int height = crop.h <= 1 ? static_cast<int>(crop.h * h_) : crop.h;
  int width = crop.w <= 1 ? static_cast<int>(crop.w * w_) : crop.w;
  im_crop->Reshape(c_, height, width, order_);

  int w_off = crop.w <= 1 ? static_cast<int>(crop.x * w_) : crop.x;
  int h_off = crop.h <= 1 ? static_cast<int>(crop.y * h_) : crop.y;
  int src_step = w_ * c_, dst_step = width * c_;
  for (int h = 0; h < im_crop->h_; ++h) {
    const auto data_src = im_src.data() + (h + h_off) * src_step + w_off * c_;
    auto data_crop = im_crop->data() + h * dst_step;
    memcpy(data_crop, data_src, dst_step * sizeof(unsigned char));
  }
}

template <typename T>
void CropResize(const JImage &im_src, JImage *im_res, const Rect<T> &crop,
                int height, int width) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_res);
  CHECK_NE(im_src.data(), im_res->data());

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  const auto &order_ = im_src.order();

  if (order_ != kGray && order_ != kRGB && order_ != kBGR) {
    LOG(FATAL) << "Unsupported format " << order_ << " to crop and resize!";
  }

  if (crop.w <= 1 && crop.h <= 1) {
    if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > 1 ||
        crop.y + crop.h > 1) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else if (crop.w > 1 && crop.h > 1) {
    if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > w_ ||
        crop.y + crop.h > h_) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else {
    LOG(FATAL) << "Crop scale must be the same!";
  }

  im_res->Reshape(c_, height, width, order_);

  const auto *data_src = im_src.data();
  auto *data_res = im_res->data();

  float step_h =
      (crop.h <= 1 ? crop.h * h_ : crop.h) / static_cast<float>(height);
  float step_w =
      (crop.w <= 1 ? crop.w * w_ : crop.w) / static_cast<float>(width);
  float h_off = crop.h <= 1 ? crop.y * h_ : crop.y;
  float w_off = crop.w <= 1 ? crop.x * w_ : crop.x;
  int s_h, s_w, src_offset, dst_offset;
  int src_step = w_ * c_, dst_step = width * c_;
  for (int c = 0; c < c_; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        s_h = static_cast<int>(h_off + step_h * h);
        s_w = static_cast<int>(w_off + step_w * w);
        src_offset = s_h * src_step + s_w * c_;
        dst_offset = h * dst_step + w * c_;
        data_res[dst_offset + c] = data_src[src_offset + c];
      }
    }
  }
}

template <typename T>
void CropResize2Gray(const JImage &im_src, JImage *im_gray, const Rect<T> &crop,
                     int height, int width) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_gray);
  CHECK_NE(im_src.data(), im_gray->data());

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  const auto &order_ = im_src.order();

  int loc_r = 0, loc_g = 1, loc_b = 2;
  if (order_ == kRGB) {
    loc_r = 0, loc_g = 1, loc_b = 2;
  } else if (order_ == kBGR) {
    loc_r = 2, loc_g = 1, loc_b = 0;
  } else if (order_ == kGray || order_ == kI420) {
    loc_r = 0, loc_g = 0, loc_b = 0, c_ = 1;
  } else {
    LOG(FATAL) << "Unsupported format " << order_
               << " to crop and resize to gray!";
  }

  if (crop.w <= 1 && crop.h <= 1) {
    if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > 1 ||
        crop.y + crop.h > 1) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else if (crop.w > 1 && crop.h > 1) {
    if (crop.x < 0 || crop.y < 0 || crop.x + crop.w > w_ ||
        crop.y + crop.h > h_) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else {
    LOG(FATAL) << "Crop scale must be the same!";
  }

  im_gray->Reshape(1, height, width, kGray);

  const auto *data_src = im_src.data();
  auto *data_gray = im_gray->data();

  float step_h =
      (crop.h <= 1 ? crop.h * h_ : crop.h) / static_cast<float>(height);
  float step_w =
      (crop.w <= 1 ? crop.w * w_ : crop.w) / static_cast<float>(width);
  float h_off = crop.h <= 1 ? crop.y * h_ : crop.y;
  float w_off = crop.w <= 1 ? crop.x * w_ : crop.x;
  float sum;
  int s_h, s_w, src_offset, src_step = w_ * c_;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      s_h = static_cast<int>(h_off + step_h * h);
      s_w = static_cast<int>(w_off + step_w * w);
      src_offset = s_h * src_step + s_w * c_;
      sum = 0.299f * data_src[src_offset + loc_r] +
            0.587f * data_src[src_offset + loc_g] +
            0.114f * data_src[src_offset + loc_b];
      *data_gray++ = static_cast<unsigned char>(sum);
    }
  }
}

// Filter, Gaussian Blur and Canny.
void Filter1D(const JImage &im_src, JImage *im_filter, const float *kernel,
              int kernel_size, int direction) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_filter);

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  const auto &order_ = im_src.order();

  im_filter->Reshape(c_, h_, w_, order_);

  const auto *data_src = im_src.data();
  auto *data_filter = im_filter->data();

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
        if (order_ != kGray) {
          val_c1 += data_src[im_index + 1] * val_kernel;
          val_c2 += data_src[im_index + 2] * val_kernel;
        }
      }
      *data_filter++ =
          (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c0));
      if (order_ != kGray) {
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
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_filter);

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  const auto &order_ = im_src.order();

  im_filter->Reshape(c_, h_, w_, order_);

  const auto *data_src = im_src.data();
  auto *data_filter = im_filter->data();

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
          if (order_ != kGray) {
            val_c1 += data_src[im_index + 1] * val_kernel;
            val_c2 += data_src[im_index + 2] * val_kernel;
          }
        }
      }
      *data_filter++ =
          (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c0));
      if (order_ != kGray) {
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
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(im_blur);

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  const auto &order_ = im_src.order();

  im_blur->Reshape(c_, h_, w_, order_);

  const auto *data_src = im_src.data();
  auto *data_blur = im_blur->data();

  auto kernel = std::vector<float>(kernel_size, 0);
  GetGaussianKernel(kernel.data(), kernel_size, sigma);

  float val_c0, val_c1, val_c2, val_kernel;
  int im_h, im_w, im_index, center = kernel_size >> 1;

  auto data_w = std::vector<float>(c_ * h_ * w_, 0);
  float *data_w_index = data_w.data();
  for (int h = 0; h < h_; ++h) {
    for (int w = 0; w < w_; ++w) {
      val_c0 = 0.f, val_c1 = 0.f, val_c2 = 0.f;
      for (int k_w = 0; k_w < kernel_size; ++k_w) {
        im_w = std::abs(w - center + k_w);
        im_w = im_w < w_ ? im_w : ((w_ << 1) - 1 - im_w) % w_;
        im_index = (w_ * h + im_w) * c_;
        val_kernel = kernel[k_w];
        val_c0 += data_src[im_index + 0] * val_kernel;
        if (order_ != kGray) {
          val_c1 += data_src[im_index + 1] * val_kernel;
          val_c2 += data_src[im_index + 2] * val_kernel;
        }
      }
      *data_w_index++ = val_c0;
      if (order_ != kGray) {
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
        if (order_ != kGray) {
          val_c1 += data_w[im_index + 1] * val_kernel;
          val_c2 += data_w[im_index + 2] * val_kernel;
        }
      }
      *data_blur++ =
          (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c0));
      if (order_ != kGray) {
        *data_blur++ =
            (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c1));
        *data_blur++ =
            (unsigned char)Util::constrain(0, 255, static_cast<int>(val_c2));
      }
    }
  }
}

void Canny(const JImage &im_src, JImage *im_canny, float thresh_low,
           float thresh_high, bool L2) {
  CHECK_NOTNULL(im_src.data());

  const auto &order = im_src.order();

  if (order == kRGB) {
    FormatTransform(im_src, im_canny, kRGB2Gray);
  } else if (order == kBGR) {
    FormatTransform(im_src, im_canny, kBGR2Gray);
  } else if (order == kI420) {
    FormatTransform(im_src, im_canny, kI4202Gray);
  } else if (order == kGray) {
    im_src.CopyTo(im_canny);
  }

  int h_ = im_canny->h_, w_ = im_canny->w_;
  auto *data_ = im_canny->data();

  auto grad_x = std::vector<int>(h_ * w_, 0);
  auto grad_y = std::vector<int>(h_ * w_, 0);
  auto magnitude = std::vector<int>(h_ * w_, 0);
  Gradient(*im_canny, grad_x.data(), grad_y.data(), magnitude.data(), L2);

  if (L2) {
    if (thresh_low > 0) thresh_low *= thresh_low;
    if (thresh_high > 0) thresh_high *= thresh_high;
  }

  // 0 - the pixel might belong to an edge
  // 1 - the pixel can not belong to an edge
  // 2 - the pixel does belong to an edge

#define CANNY_SHIFT 15
  const auto TG22 = static_cast<int>(
      0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) + 0.5);

  memset(im_canny->data(), 0, h_ * w_ * sizeof(unsigned char));
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

  auto *data_index = data_;
  for (int i = 0; i < h_ * w_; ++i, ++data_index) {
    *data_index = (unsigned char)-(*data_index >> 1);
  }
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
          : nullptr;

  float sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5f - 1) * 0.3f + 0.8f;
  float scale2X = -0.5f / (sigmaX * sigmaX);

  float sum = 0;
  for (int i = 0; i < n; ++i) {
    float x = i - (n - 1) * 0.5f;
    float t =
        fixed_kernel != nullptr ? fixed_kernel[i] : std::exp(scale2X * x * x);
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
  CHECK_NOTNULL(im_src.data());

  const auto *data_src = im_src.data();
  int h_ = im_src.h_, w_ = im_src.w_;

  JImage *im_gray = nullptr;
  if (im_src.order() != kGray) {
    im_gray = new JImage();
    im_src.CopyTo(im_gray);
    data_src = im_gray->data();
  }

  memset(grad_x, 0, h_ * w_ * sizeof(int));
  memset(grad_y, 0, h_ * w_ * sizeof(int));
  memset(magnitude, 0, h_ * w_ * sizeof(int));

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

// Explicit instantiation
template void Line(JImage *im, const PointI &start, const PointI &end,
                   const Scalar &scalar);
template void Line(JImage *im, const PointF &start, const PointF &end,
                   const Scalar &scalar);

template void Rectangle(JImage *im, const RectI &rect, const Scalar &scalar);
template void Rectangle(JImage *im, const RectF &rect, const Scalar &scalar);

template void Crop(const JImage &im_src, JImage *im_crop, const RectI &crop);
template void Crop(const JImage &im_src, JImage *im_crop, const RectF &crop);

template void CropResize(const JImage &im_src, JImage *im_res,
                         const RectI &crop, int height, int width);
template void CropResize(const JImage &im_src, JImage *im_res,
                         const RectF &crop, int height, int width);

template void CropResize2Gray(const JImage &im_src, JImage *im_gray,
                              const RectI &crop, int height, int width);
template void CropResize2Gray(const JImage &im_src, JImage *im_gray,
                              const RectF &crop, int height, int width);

}  // namespace JImageProc

}  // namespace Shadow
