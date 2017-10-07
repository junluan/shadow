#ifndef SHADOW_EXAMPLES_DETECTION_MTCNN_HPP
#define SHADOW_EXAMPLES_DETECTION_MTCNN_HPP

#include "method.hpp"

namespace Shadow {

class DetectionMTCNN final : public Method {
 public:
  DetectionMTCNN() = default;
  ~DetectionMTCNN() override { Release(); }

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
  void Process_net_12(const float *data, const VecInt &in_shape, float height,
                      float width, float threshold, float scale,
                      VecBoxF *boxes);
  void Process_net_24(const float *data, const VecInt &in_shape, float height,
                      float width, float threshold, const VecBoxF &net_12_boxes,
                      VecBoxF *boxes);
  void Process_net_48(const float *data, const VecInt &in_shape, float height,
                      float width, float threshold, const VecBoxF &net_24_boxes,
                      VecBoxF *boxes);

  void CalculateScales(float height, float width, float factor, float max_side,
                       float min_side, VecFloat *scales);

  BoxF Rect2SquareWithConstrain(const BoxF &box, float height, float width);

  Network net_12_, net_24_, net_48_;
  VecFloat net_12_in_data_, net_24_in_data_, net_48_in_data_, thresholds_,
      nms_thresholds_, scales_;
  VecInt net_12_in_shape_, net_24_in_shape_, net_48_in_shape_;
  VecBoxF net_12_boxes_, net_24_boxes_, net_48_boxes_;
  int net_24_in_c_, net_24_in_h_, net_24_in_w_, net_24_in_num_;
  int net_48_in_c_, net_48_in_h_, net_48_in_w_, net_48_in_num_;
  float factor_, max_side_, min_side_;
  JImage im_ini_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DETECTION_MTCNN_HPP
