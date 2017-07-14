#ifndef SHADOW_EXAMPLES_DETECTION_SSD_HPP
#define SHADOW_EXAMPLES_DETECTION_SSD_HPP

#include "method.hpp"

namespace Shadow {

using LabelBBox = std::map<int, VecBoxF>;
using VecLabelBBox = std::vector<LabelBBox>;

class DetectionSSD final : public Method {
 public:
  DetectionSSD() {}
  ~DetectionSSD() { Release(); }

  virtual void Setup(const std::string &model_file, int classes,
                     int batch) override;

  virtual void Predict(const JImage &im_src, const VecRectF &rois,
                       std::vector<VecBoxF> *Bboxes) override;
#if defined(USE_OpenCV)
  virtual void Predict(const cv::Mat &im_mat, const VecRectF &rois,
                       std::vector<VecBoxF> *Bboxes) override;
#endif

  virtual void Release() override;

 private:
  void Process(const float *data, std::vector<VecBoxF> *Bboxes);

  void GetLocPredictions(const float *loc_data, int num,
                         int num_preds_per_class, int num_loc_classes,
                         bool share_location, VecLabelBBox *loc_preds);
  void GetConfidenceScores(const float *conf_data, int num,
                           int num_preds_per_class, int num_classes,
                           std::vector<std::map<int, VecFloat>> *conf_preds);
  void GetPriorBBoxes(const float *prior_data, int num_priors,
                      VecBoxF *prior_bboxes,
                      std::vector<VecFloat> *prior_variances);

  void DecodeBBoxesAll(const VecLabelBBox &all_loc_preds,
                       const VecBoxF &prior_bboxes,
                       const std::vector<VecFloat> &prior_variances, int num,
                       bool share_location, int num_loc_classes,
                       int background_label_id,
                       VecLabelBBox *all_decode_bboxes);
  void DecodeBBoxes(const VecBoxF &prior_bboxes,
                    const std::vector<VecFloat> &prior_variances,
                    const VecBoxF &bboxes, VecBoxF *decode_bboxes);
  void DecodeBBox(const BoxF &prior_bbox, const VecFloat &prior_variance,
                  const BoxF &bbox, BoxF *decode_bbox);

  Network net_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
  int num_classes_, num_priors_, num_loc_classes_, background_label_id_, top_k_,
      keep_top_k_;
  float threshold_, nms_threshold_, confidence_threshold_;
  bool share_location_;
  JImage im_ini_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DETECTION_SSD_HPP
