#ifndef SHADOW_EXAMPLES_METHOD_HPP
#define SHADOW_EXAMPLES_METHOD_HPP

#include "shadow/network.hpp"
#include "shadow/util/boxes.hpp"
#include "shadow/util/jimage.hpp"
#include "shadow/util/util.hpp"

class Method {
 public:
  Method() {}
  virtual ~Method() {}

  virtual void Setup(const std::string &model_file, int batch = 1) {
    LOG(INFO) << "Setup method!";
  }

  virtual void Predict(const JImage &image, const VecRectF &rois,
                       std::vector<VecBoxF> *Bboxes) {
    LOG(INFO) << "Predict for JImage!";
  }
#if defined(USE_OpenCV)
  virtual void Predict(const cv::Mat &im_mat, const VecRectF &rois,
                       std::vector<VecBoxF> *Bboxes) {
    LOG(INFO) << "Predict for Mat!";
  }
#endif

  virtual void Release() { LOG(INFO) << "Release method!"; }
};

#endif  // SHADOW_EXAMPLES_METHOD_HPP
