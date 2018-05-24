#ifndef SHADOW_EXAMPLES_DETECTION_REFINEDET_HPP
#define SHADOW_EXAMPLES_DETECTION_REFINEDET_HPP

#include "method.hpp"

namespace Shadow {

class DetectionRefineDet final : public Method {
 public:
  DetectionRefineDet() = default;
  ~DetectionRefineDet() override { Release(); }

  void Setup(const VecString &model_files, const VecInt &in_shape) override;

  void Predict(const JImage &im_src, const VecRectF &rois,
               std::vector<VecBoxF> *Gboxes,
               std::vector<std::vector<VecPointF>> *Gpoints) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<VecBoxF> *Gboxes,
               std::vector<std::vector<VecPointF>> *Gpoints) override;
#endif

  void Release() override;

 private:
  using LabelBBox = std::map<int, VecBoxF>;
  using VecLabelBBox = std::vector<LabelBBox>;

  void Process(const VecFloat &in_data, std::vector<VecBoxF> *Gboxes);

  void GetLocPredictions(const float *loc_data, int num,
                         int num_preds_per_class, int num_loc_classes,
                         bool share_location, VecLabelBBox *loc_preds);
  void OSGetConfidenceScores(const float *conf_data, const float *arm_conf_data,
                             int num, int num_preds_per_class, int num_classes,
                             std::vector<std::map<int, VecFloat>> *conf_preds,
                             float objectness_score);
  void GetPriorBBoxes(const float *prior_data, int num_priors,
                      VecBoxF *prior_bboxes,
                      std::vector<VecFloat> *prior_variances);

  void CasRegDecodeBBoxesAll(const VecLabelBBox &all_loc_preds,
                             const VecBoxF &prior_bboxes,
                             const std::vector<VecFloat> &prior_variances,
                             int num, bool share_location, int num_loc_classes,
                             int background_label_id,
                             VecLabelBBox *all_decode_bboxes,
                             const VecLabelBBox &all_arm_loc_preds);
  void DecodeBBoxes(const VecBoxF &prior_bboxes,
                    const std::vector<VecFloat> &prior_variances,
                    const VecBoxF &bboxes, VecBoxF *decode_bboxes);
  void DecodeBBox(const BoxF &prior_bbox, const VecFloat &prior_variance,
                  const BoxF &bbox, BoxF *decode_bbox);

  Network net_;
  VecFloat in_data_;
  std::string in_str_, odm_loc_str_, odm_conf_flatten_str_, arm_priorbox_str_,
      arm_conf_flatten_str_, arm_loc_str_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
  int num_classes_, num_priors_, num_loc_classes_, background_label_id_, top_k_,
      keep_top_k_;
  float threshold_, nms_threshold_, confidence_threshold_, objectness_score_;
  bool share_location_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DETECTION_REFINEDET_HPP
