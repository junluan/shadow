#ifndef SHADOW_EXAMPLES_CLASSIFICATION_HPP
#define SHADOW_EXAMPLES_CLASSIFICATION_HPP

#include "method.hpp"

namespace Shadow {

class Classification final : public Method {
 public:
  Classification() = default;

  void Setup(const std::string &model_file) override;

  void Predict(const JImage &im_src, const RectF &roi,
               std::map<std::string, VecFloat> *scores) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const RectF &roi,
               std::map<std::string, VecFloat> *scores) override;
#endif

 private:
  void Process(const VecFloat &in_data,
               std::map<std::string, VecFloat> *scores);

  Network net_;
  VecFloat in_data_;
  std::string in_str_, prob_str_;
  int num_classes_, batch_, in_num_, in_c_, in_h_, in_w_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_CLASSIFICATION_HPP
