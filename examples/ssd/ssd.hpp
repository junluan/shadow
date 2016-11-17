#ifndef SHADOW_SSD_HPP
#define SHADOW_SSD_HPP

#include "shadow/network.hpp"
#include "shadow/util/boxes.hpp"
#include "shadow/util/jimage.hpp"
#include "shadow/util/util.hpp"

typedef std::map<int, VecBoxF> LabelBBox;
typedef std::vector<LabelBBox> VecLabelBBox;

class SSD {
 public:
  void Setup(const std::string &model_file, int batch = 1);
  void Predict(const JImage &image, const VecRectF &rois,
               std::vector<VecBoxF> *Bboxes);
#if defined(USE_ArcSoft)
  void Predict(const ASVLOFFSCREEN &im_arc, const VecRectF &rois,
               std::vector<VecBoxF> *Bboxes);
#endif
  void Release();

 private:
  void Process(const float *data, std::vector<VecBoxF> *Bboxes);

  const float *GetBlobDataByName(const std::string &name);
  const Blob<float> *GetBlobByName(const std::string &name);

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
  float *in_data_;
  std::map<std::string, float *> blobs_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
  int num_classes_, num_priors_, num_loc_classes_, background_label_id_, top_k_,
      keep_top_k_;
  float threshold_, nms_threshold_, confidence_threshold_;
  bool share_location_;
  JImage im_res_;
};

#endif  // SHADOW_SSD_HPP
