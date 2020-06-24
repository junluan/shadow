#ifndef SHADOW_ALGORITHM_CLASSIFY_HPP_
#define SHADOW_ALGORITHM_CLASSIFY_HPP_

#include "method.hpp"

#include "core/network.hpp"

namespace Shadow {

class Classify final : public Method {
 public:
  Classify() = default;

  void Setup(const std::string& model_file) override;

  void Predict(const JImage& im_src, const RectF& roi,
               std::map<std::string, VecFloat>* scores) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat& im_mat, const RectF& roi,
               std::map<std::string, VecFloat>* scores) override;
#endif

 private:
  void Process(const VecFloat& in_data,
               std::map<std::string, VecFloat>* scores);

  Network net_;
  VecFloat in_data_;
  std::string in_str_, prob_str_;
  int num_classes_, batch_, in_num_, in_c_, in_h_, in_w_;
};

}  // namespace Shadow

#endif  // SHADOW_ALGORITHM_CLASSIFY_HPP_
