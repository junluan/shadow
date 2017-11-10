#ifndef SHADOW_EXAMPLES_METHOD_HPP
#define SHADOW_EXAMPLES_METHOD_HPP

#include "core/network.hpp"
#include "util/boxes.hpp"
#include "util/jimage.hpp"
#include "util/util.hpp"

namespace Shadow {

class Method {
 public:
  Method() = default;
  virtual ~Method() = default;

  virtual void Setup(const VecString &model_files, const VecInt &in_shape) {
    LOG(FATAL) << "Setup method!";
  }

  virtual void Predict(const JImage &im_src, const VecRectF &rois,
                       std::vector<VecBoxF> *Gboxes,
                       std::vector<std::vector<VecPointF>> *Gpoints) {
    LOG(FATAL) << "Predict for JImage!";
  }
  virtual void Predict(const JImage &im_src, const VecRectF &rois,
                       std::vector<std::map<std::string, VecFloat>> *scores) {
    LOG(FATAL) << "Predict for JImage!";
  }

#if defined(USE_OpenCV)
  virtual void Predict(const cv::Mat &im_mat, const VecRectF &rois,
                       std::vector<VecBoxF> *Gboxes,
                       std::vector<std::vector<VecPointF>> *Gpoints) {
    LOG(FATAL) << "Predict for Mat!";
  }
  virtual void Predict(const cv::Mat &im_mat, const VecRectF &rois,
                       std::vector<std::map<std::string, VecFloat>> *scores) {
    LOG(FATAL) << "Predict for Mat!";
  }
#endif

  virtual void Release() { LOG(FATAL) << "Release method!"; }
};

static inline void ConvertData(const JImage &im_src, float *data,
                               const RectF &roi, int channel, int height,
                               int width, int flag = 1,
                               bool transpose = false) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(data);

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  int dst_spatial_dim = height * width;
  const auto &order_ = im_src.order();

  int loc_r = 0, loc_g = 1, loc_b = 2;
  if (order_ == kGray) {
    loc_r = loc_g = loc_b = 0;
  } else if (order_ == kRGB) {
    loc_r = 0, loc_g = 1, loc_b = 2;
  } else if (order_ == kBGR) {
    loc_r = 2, loc_g = 1, loc_b = 0;
  } else {
    LOG(FATAL) << "Unsupported format " << order_
               << " to crop and resize to gray!";
  }

  if (roi.w <= 1 && roi.h <= 1) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > 1 || roi.y + roi.h > 1) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else if (roi.w > 1 && roi.h > 1) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > w_ || roi.y + roi.h > h_) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else {
    LOG(FATAL) << "Crop scale must be the same!";
  }

  const auto *data_src = im_src.data();
  float *data_r = nullptr, *data_g = nullptr, *data_b = nullptr,
        *data_gray = nullptr;
  if (channel == 3 && flag == 0) {
    // Convert to RRRGGGBBB
    CHECK((order_ != kGray));
    data_r = data;
    data_g = data + dst_spatial_dim;
    data_b = data + (dst_spatial_dim << 1);
  } else if (channel == 3 && flag == 1) {
    // Convert to BBBGGGRRR
    CHECK((order_ != kGray));
    data_r = data + (dst_spatial_dim << 1);
    data_g = data + dst_spatial_dim;
    data_b = data;
  } else if (channel == 1) {
    // Convert to Gray
    data_gray = data;
  } else {
    LOG(FATAL) << "Unsupported flag " << flag;
  }

  float step_h = (roi.h <= 1 ? roi.h * h_ : roi.h) / static_cast<float>(height);
  float step_w = (roi.w <= 1 ? roi.w * w_ : roi.w) / static_cast<float>(width);
  float h_off = roi.h <= 1 ? roi.y * h_ : roi.y;
  float w_off = roi.w <= 1 ? roi.x * w_ : roi.x;
  int s_h, s_w, src_offset, dst_offset;
  int src_step = w_ * c_;
  if (channel == 1) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        s_h = static_cast<int>(h_off + step_h * h);
        s_w = static_cast<int>(w_off + step_w * w);
        src_offset = s_h * src_step + s_w * c_;
        dst_offset = h * width + w;
        data_gray[dst_offset] = 0.299f * data_src[src_offset + loc_r];
        data_gray[dst_offset] += 0.587f * data_src[src_offset + loc_g];
        data_gray[dst_offset] += 0.114f * data_src[src_offset + loc_b];
      }
    }
  } else {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        s_h = static_cast<int>(h_off + step_h * h);
        s_w = static_cast<int>(w_off + step_w * w);
        src_offset = s_h * src_step + s_w * c_;
        if (transpose) {
          dst_offset = w * height + h;
        } else {
          dst_offset = h * width + w;
        }
        data_r[dst_offset] = data_src[src_offset + loc_r];
        data_g[dst_offset] = data_src[src_offset + loc_g];
        data_b[dst_offset] = data_src[src_offset + loc_b];
      }
    }
  }
}

#if defined(USE_OpenCV)
static inline void ConvertData(const cv::Mat &im_mat, float *data,
                               const RectF &roi, int channel, int height,
                               int width, int flag = 1,
                               bool transpose = false) {
  CHECK(!im_mat.empty());
  CHECK_NOTNULL(data);

  int c_ = im_mat.channels(), h_ = im_mat.rows, w_ = im_mat.cols;
  int dst_spatial_dim = height * width;

  if (roi.w <= 1 && roi.h <= 1) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > 1 || roi.y + roi.h > 1) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else if (roi.w > 1 && roi.h > 1) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > w_ || roi.y + roi.h > h_) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else {
    LOG(FATAL) << "Crop scale must be the same!";
  }

  float *data_r = nullptr, *data_g = nullptr, *data_b = nullptr,
        *data_gray = nullptr;
  if (channel == 3 && flag == 0) {
    // Convert to RRRGGGBBB
    data_r = data;
    data_g = data + dst_spatial_dim;
    data_b = data + (dst_spatial_dim << 1);
  } else if (channel == 3 && flag == 1) {
    // Convert to BBBGGGRRR
    data_r = data + (dst_spatial_dim << 1);
    data_g = data + dst_spatial_dim;
    data_b = data;
  } else if (channel == 1) {
    // Convert to Gray
    data_gray = data;
  } else {
    LOG(FATAL) << "Unsupported flag " << flag;
  }

  auto roi_x = static_cast<int>(roi.w <= 1 ? roi.x * w_ : roi.x);
  auto roi_y = static_cast<int>(roi.h <= 1 ? roi.y * h_ : roi.y);
  auto roi_w = static_cast<int>(roi.w <= 1 ? roi.w * w_ : roi.w);
  auto roi_h = static_cast<int>(roi.h <= 1 ? roi.h * h_ : roi.h);

  cv::Rect cv_roi(roi_x, roi_y, roi_w, roi_h);
  cv::Size cv_size(width, height);

  cv::Mat im_resize;
  if (roi_x != 0 || roi_y != 0 || roi_w != w_ || roi_h != h_) {
    cv::resize(im_mat(cv_roi), im_resize, cv_size);
  } else {
    cv::resize(im_mat, im_resize, cv_size);
  }

  int dst_h = height, dst_w = width;
  if (transpose) {
    cv::transpose(im_resize, im_resize);
    dst_h = width, dst_w = height;
  }

  if (channel == 3) {
    CHECK_EQ(c_, 3);
    for (int h = 0; h < dst_h; ++h) {
      const auto *data_src = im_resize.ptr<uchar>(h);
      for (int w = 0; w < dst_w; ++w) {
        *data_b++ = static_cast<float>(*data_src++);
        *data_g++ = static_cast<float>(*data_src++);
        *data_r++ = static_cast<float>(*data_src++);
      }
    }
  } else if (channel == 1) {
    cv::Mat im_gray;
    cv::cvtColor(im_resize, im_gray, cv::COLOR_BGR2GRAY);
    for (int h = 0; h < dst_h; ++h) {
      const auto *data_src = im_gray.ptr<uchar>(h);
      for (int w = 0; w < dst_w; ++w) {
        *data_gray++ = static_cast<float>(*data_src++);
      }
    }
  } else {
    LOG(FATAL) << "Unsupported flag " << flag;
  }
}
#endif

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_METHOD_HPP
