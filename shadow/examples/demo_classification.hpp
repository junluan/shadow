#ifndef SHADOW_EXAMPLES_DEMO_CLASSIFICATION_HPP
#define SHADOW_EXAMPLES_DEMO_CLASSIFICATION_HPP

#include "classification.hpp"

#include <memory>

namespace Shadow {

class DemoClassification {
 public:
  explicit DemoClassification(
      const std::string &method_name = "classification") {
    if (method_name == "classification") {
      method_ = std::make_shared<Classification>();
    } else {
      LOG(FATAL) << "Unknown method " << method_name;
    }
  }

  void Setup(const std::string &model_file) { method_->Setup(model_file); }

  void Test(const std::string &image_file);
  void BatchTest(const std::string &list_file);

  void Predict(const JImage &im_src, const RectF &roi,
               std::map<std::string, VecFloat> *scores) {
    method_->Predict(im_src, roi, scores);
  }
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const RectF &roi,
               std::map<std::string, VecFloat> *scores) {
    method_->Predict(im_mat, roi, scores);
  }
#endif

 private:
  void PrintDetections(const std::string &im_name,
                       const std::map<std::string, VecFloat> &scores,
                       std::ostream *os);

  std::shared_ptr<Method> method_ = nullptr;
  JImage im_ini_;
  std::map<std::string, VecFloat> scores_;
  Timer timer_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DEMO_CLASSIFICATION_HPP
