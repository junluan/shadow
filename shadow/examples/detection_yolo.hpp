#ifndef SHADOW_EXAMPLES_DETECTION_YOLO_HPP
#define SHADOW_EXAMPLES_DETECTION_YOLO_HPP

#include "method.hpp"

namespace Shadow {

class DetectionYOLO final : public Method {
 public:
  DetectionYOLO() = default;

  void Setup(const std::string &model_file) override;

  void Predict(const JImage &im_src, const VecRectF &rois,
               std::vector<VecBoxF> *Gboxes,
               std::vector<std::vector<VecPointF>> *Gpoints) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<VecBoxF> *Gboxes,
               std::vector<std::vector<VecPointF>> *Gpoints) override;
#endif

 private:
  void Process(const VecFloat &in_data, std::vector<VecBoxF> *Gboxes);

  void ConvertDetections(float *data, float *biases, int classes, int num_km,
                         int side, float threshold, VecBoxF *boxes);

  Network net_;
  VecFloat in_data_, biases_;
  std::string in_str_, out_str_;
  int batch_, in_num_, in_c_, in_h_, in_w_, out_num_, out_hw_;
  int num_classes_, num_km_;
  float threshold_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DETECTION_YOLO_HPP
