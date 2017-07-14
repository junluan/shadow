#ifndef SHADOW_EXAMPLES_METHOD_HPP
#define SHADOW_EXAMPLES_METHOD_HPP

#include "core/network.hpp"
#include "util/boxes.hpp"
#include "util/jimage.hpp"
#include "util/util.hpp"

namespace Shadow {

class Method {
 public:
  Method() {}
  virtual ~Method() {}

  virtual void Setup(const std::string &model_file, int classes, int batch) {
    LOG(INFO) << "Setup method!";
  }

  virtual void Predict(const JImage &im_src, const VecRectF &rois,
                       std::vector<VecBoxF> *Bboxes) {
    LOG(INFO) << "Predict for JImage!";
  }
  virtual void Predict(const JImage &im_src, const VecRectF &rois,
                       std::vector<std::map<std::string, VecFloat>> *scores) {
    LOG(INFO) << "Predict for JImage!";
  }

#if defined(USE_OpenCV)
  virtual void Predict(const cv::Mat &im_mat, const VecRectF &rois,
                       std::vector<VecBoxF> *Bboxes) {
    LOG(INFO) << "Predict for Mat!";
  }
  virtual void Predict(const cv::Mat &im_mat, const VecRectF &rois,
                       std::vector<std::map<std::string, VecFloat>> *scores) {
    LOG(INFO) << "Predict for Mat!";
  }
#endif

  virtual void Release() { LOG(INFO) << "Release method!"; }
};

inline void ConvertData(const JImage &im_src, float *data, const RectF &roi,
                        int height, int width, int flag = 1) {
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
  float *data_r = nullptr, *data_g = nullptr, *data_b = nullptr;
  if (order_ == kGray) {
    // Convert to Gray
    CHECK_EQ(c_, 1);
    data_r = data_g = data_b = data;
  } else if (flag == 0) {
    // Convert to RRRGGGBBB
    data_r = data;
    data_g = data + dst_spatial_dim;
    data_b = data + (dst_spatial_dim << 1);
  } else if (flag == 1) {
    // Convert to BBBGGGRRR
    data_r = data + (dst_spatial_dim << 1);
    data_g = data + dst_spatial_dim;
    data_b = data;
  } else {
    LOG(FATAL) << "Unsupported flag " << flag;
  }

  float step_h = (roi.h <= 1 ? roi.h * h_ : roi.h) / static_cast<float>(height);
  float step_w = (roi.w <= 1 ? roi.w * w_ : roi.w) / static_cast<float>(width);
  float h_off = roi.h <= 1 ? roi.y * h_ : roi.y;
  float w_off = roi.w <= 1 ? roi.x * w_ : roi.x;
  int s_h, s_w, src_offset, dst_offset;
  int src_step = w_ * c_;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      s_h = static_cast<int>(h_off + step_h * h);
      s_w = static_cast<int>(w_off + step_w * w);
      src_offset = s_h * src_step + s_w * c_;
      dst_offset = h * width + w;
      data_r[dst_offset] = data_src[src_offset + loc_r];
      data_g[dst_offset] = data_src[src_offset + loc_g];
      data_b[dst_offset] = data_src[src_offset + loc_b];
    }
  }
}

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_METHOD_HPP
