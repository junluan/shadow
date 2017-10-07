#ifndef SHADOW_EXAMPLES_DETECTION_YOLO_HPP
#define SHADOW_EXAMPLES_DETECTION_YOLO_HPP

#include "method.hpp"

namespace Shadow {

class DetectionYOLO final : public Method {
 public:
  DetectionYOLO() = default;
  ~DetectionYOLO() override { Release(); }

  void Setup(const VecString &model_files, const VecInt &classes,
             const VecInt &in_shape) override;

  void Predict(const JImage &im_src, const VecRectF &rois,
               std::vector<VecBoxF> *Bboxes) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<VecBoxF> *Bboxes) override;
#endif

  void Release() override;

 private:
  void Process(const float *data, std::vector<VecBoxF> *Bboxes);

  void ConvertDetections(float *data, float *biases, int classes, int num_km,
                         int side, float threshold, VecBoxF *boxes);

  Network net_;
  VecFloat in_data_, out_data_, biases_;
  int batch_, in_num_, in_c_, in_h_, in_w_, out_num_, out_hw_;
  int num_classes_, num_km_;
  float threshold_;
  JImage im_ini_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DETECTION_YOLO_HPP
