#ifndef SHADOW_ALGORITHM_DETECT_FASTER_RCNN_HPP_
#define SHADOW_ALGORITHM_DETECT_FASTER_RCNN_HPP_

#include "method.hpp"

#include "core/network.hpp"

namespace Shadow {

class DetectFasterRCNN final : public Method {
 public:
  DetectFasterRCNN() = default;

  void Setup(const std::string& model_file) override;

  void Predict(const JImage& im_src, const RectF& roi, VecBoxF* boxes,
               std::vector<VecPointF>* Gpoints) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat& im_mat, const RectF& roi, VecBoxF* boxes,
               std::vector<VecPointF>* Gpoints) override;
#endif

 private:
  void Process(const VecFloat& in_data, const VecInt& in_shape,
               const VecFloat& im_info, float height, float width,
               VecBoxF* boxes);

  void CalculateScales(float height, float width, float max_side,
                       const VecFloat& min_side, VecFloat* scales);

  Network net_;
  VecFloat in_data_, min_side_, scales_, im_info_;
  VecInt in_shape_;
  std::string in_str_, im_info_str_, rois_str_, bbox_pred_str_, cls_prob_str_;
  int num_classes_;
  float max_side_, threshold_, nms_threshold_;
  bool is_bgr_, class_agnostic_;
};

}  // namespace Shadow

#endif  // SHADOW_ALGORITHM_DETECT_FASTER_RCNN_HPP_
