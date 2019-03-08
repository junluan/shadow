#ifndef SHADOW_ALGORITHM_DETECT_SSD_HPP
#define SHADOW_ALGORITHM_DETECT_SSD_HPP

#include "method.hpp"

#include "core/network.hpp"

namespace Shadow {

class DetectSSD final : public Method {
 public:
  DetectSSD() = default;

  void Setup(const std::string &model_file) override;

  void Predict(const JImage &im_src, const RectF &roi, VecBoxF *boxes,
               std::vector<VecPointF> *Gpoints) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const RectF &roi, VecBoxF *boxes,
               std::vector<VecPointF> *Gpoints) override;
#endif

 private:
  void Process(const VecFloat &in_data, std::vector<VecBoxF> *Gboxes);

  Network net_;
  VecFloat in_data_;
  std::string in_str_, out_str_;
  int batch_, in_num_, in_c_, in_h_, in_w_, background_label_id_;
  float threshold_, nms_threshold_;
};

}  // namespace Shadow

#endif  // SHADOW_ALGORITHM_DETECT_SSD_HPP
