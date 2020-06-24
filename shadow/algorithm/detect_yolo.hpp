#ifndef SHADOW_ALGORITHM_DETECT_YOLO_HPP_
#define SHADOW_ALGORITHM_DETECT_YOLO_HPP_

#include "method.hpp"

#include "core/network.hpp"

namespace Shadow {

class DetectYOLO final : public Method {
 public:
  DetectYOLO() = default;

  void Setup(const std::string& model_file) override;

  void Predict(const JImage& im_src, const RectF& roi, VecBoxF* boxes,
               std::vector<VecPointF>* Gpoints) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat& im_mat, const RectF& roi, VecBoxF* boxes,
               std::vector<VecPointF>* Gpoints) override;
#endif

 private:
  void Process(const VecFloat& in_data, std::vector<VecBoxF>* Gboxes);

  void ConvertDetections(float* data, const float* biases, int out_h, int out_w,
                         VecBoxF* boxes);

  Network net_;
  VecFloat in_data_;
  VecFloat biases_;
  std::string in_str_;
  VecString out_str_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
  int num_classes_, num_km_, version_;
  float threshold_, nms_threshold_;
};

}  // namespace Shadow

#endif  // SHADOW_ALGORITHM_DETECT_YOLO_HPP_
