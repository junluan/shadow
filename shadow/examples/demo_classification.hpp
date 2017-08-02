#ifndef SHADOW_EXAMPLES_DEMO_CLASSIFICATION_HPP
#define SHADOW_EXAMPLES_DEMO_CLASSIFICATION_HPP

#include "classification.hpp"

namespace Shadow {

class DemoClassification {
 public:
  DemoClassification(const std::string &method_name = "classification") {
    if (method_name == "classification") {
      method_ = new Classification();
    } else {
      LOG(FATAL) << "Unknown method " << method_name;
    }
  }
  ~DemoClassification() { Release(); }

  void Setup(const std::string &model_file, const VecInt &classes,
             int batch = 1) {
    method_->Setup(model_file, classes, batch);
  }
  void Release() {
    if (method_ != nullptr) {
      delete method_;
      method_ = nullptr;
    }
  }

  void Test(const std::string &image_file);
  void BatchTest(const std::string &list_file);

  void Predict(const JImage &im_src, const VecRectF &rois,
               std::vector<std::map<std::string, VecFloat>> *scores) {
    method_->Predict(im_src, rois, scores);
  }

 private:
  void PrintDetections(
      const std::string &im_name,
      const std::vector<std::map<std::string, VecFloat>> &scores,
      std::ostream *os);

  Method *method_;
  JImage im_ini_;
  std::vector<std::map<std::string, VecFloat>> scores_;
  Timer timer_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DEMO_CLASSIFICATION_HPP
