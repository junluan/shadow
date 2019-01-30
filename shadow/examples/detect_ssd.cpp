#include "detect_ssd.hpp"

namespace Shadow {

template <typename T>
inline bool SortScorePairDescend(const std::pair<float, T> &pair1,
                                 const std::pair<float, T> &pair2) {
  return pair1.first > pair2.first;
}

inline void GetMaxScoreIndex(
    const VecFloat &scores, float threshold, int top_k,
    std::vector<std::pair<float, int>> *score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

inline void ApplyNMSFast(const VecBoxF &bboxes, const VecFloat &scores,
                         float score_threshold, float nms_threshold, int top_k,
                         VecInt *indices) {
  // Sanity check.
  CHECK_EQ(bboxes.size(), scores.size());

  // Get top_k scores (with corresponding indices).
  std::vector<std::pair<float, int>> score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

  // Do nms.
  indices->clear();
  while (!score_index_vec.empty()) {
    int idx = score_index_vec.front().second;
    bool keep = true;
    for (auto id : *indices) {
      if (keep) {
        float overlap = Boxes::IoU(bboxes[idx], bboxes[id]);
        keep = overlap <= nms_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
  }
}

void DetectSSD::Setup(const std::string &model_file) {
  net_.Setup();

#if defined(USE_Protobuf)
  shadow::MetaNetParam meta_net_param;
  CHECK(IO::ReadProtoFromBinaryFile(model_file, &meta_net_param))
      << "Error when loading proto binary file: " << model_file;

  net_.LoadModel(meta_net_param.network(0));

#else
  LOG(FATAL) << "Unsupported load binary model, recompiled with USE_Protobuf";
#endif

  const auto &in_blob = net_.in_blob();
  CHECK_EQ(in_blob.size(), 1);
  in_str_ = in_blob[0];

  const auto &out_blob = net_.out_blob();
  CHECK_EQ(out_blob.size(), 3);
  mbox_loc_str_ = out_blob[0];
  mbox_conf_flatten_str_ = out_blob[1];
  mbox_priorbox_str_ = out_blob[2];

  const auto &data_shape = net_.GetBlobShapeByName<float>(in_str_);
  CHECK_EQ(data_shape.size(), 4);

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  threshold_ = net_.get_single_argument<float>("threshold", 0.6);
  num_classes_ = net_.get_single_argument<int>("num_classes", 21);
  num_loc_classes_ = 1;
  background_label_id_ = 0;
  top_k_ = net_.get_single_argument<int>("top_k", 1000);
  keep_top_k_ = net_.get_single_argument<int>("keep_top_k", 300);
  nms_threshold_ = net_.get_single_argument<float>("nms_threshold", 0.3);
  confidence_threshold_ =
      net_.get_single_argument<float>("confidence_threshold", 0.01);
  share_location_ = true;
}

void DetectSSD::Predict(const JImage &im_src, const RectF &roi, VecBoxF *boxes,
                        std::vector<VecPointF> *Gpoints) {
  ConvertData(im_src, in_data_.data(), roi, in_c_, in_h_, in_w_);

  std::vector<VecBoxF> Gboxes;
  Process(in_data_, &Gboxes);

  float height = roi.h, width = roi.w;
  for (auto &box : Gboxes[0]) {
    box.xmin *= width;
    box.xmax *= width;
    box.ymin *= height;
    box.ymax *= height;
  }

  *boxes = Gboxes[0];
}

#if defined(USE_OpenCV)
void DetectSSD::Predict(const cv::Mat &im_mat, const RectF &roi, VecBoxF *boxes,
                        std::vector<VecPointF> *Gpoints) {
  ConvertData(im_mat, in_data_.data(), roi, in_c_, in_h_, in_w_);

  std::vector<VecBoxF> Gboxes;
  Process(in_data_, &Gboxes);

  float height = roi.h, width = roi.w;
  for (auto &box : Gboxes[0]) {
    box.xmin *= width;
    box.xmax *= width;
    box.ymin *= height;
    box.ymax *= height;
  }

  *boxes = Gboxes[0];
}
#endif

void DetectSSD::Process(const VecFloat &in_data, std::vector<VecBoxF> *Gboxes) {
  std::map<std::string, float *> data_map;
  data_map[in_str_] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  const auto &prior_shape = net_.GetBlobShapeByName<float>(mbox_priorbox_str_);

  const auto *prior_data = net_.GetBlobDataByName<float>(mbox_priorbox_str_);
  const auto *loc_data = net_.GetBlobDataByName<float>(mbox_loc_str_);
  const auto *conf_data = net_.GetBlobDataByName<float>(mbox_conf_flatten_str_);

  int num_priors = prior_shape[2] / 4;

  VecLabelBBox all_loc_preds;
  GetLocPredictions(loc_data, batch_, num_priors, num_loc_classes_,
                    share_location_, &all_loc_preds);

  std::vector<std::map<int, VecFloat>> all_conf_scores;
  GetConfidenceScores(conf_data, batch_, num_priors, num_classes_,
                      &all_conf_scores);

  VecBoxF prior_bboxes;
  std::vector<VecFloat> prior_variances;
  GetPriorBBoxes(prior_data, num_priors, &prior_bboxes, &prior_variances);

  VecLabelBBox all_decode_bboxes;
  DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, batch_,
                  share_location_, num_loc_classes_, background_label_id_,
                  &all_decode_bboxes);

  int num_kept = 0;
  std::vector<std::map<int, VecInt>> all_indices;
  for (int b = 0; b < batch_; ++b) {
    const auto &conf_scores = all_conf_scores[b];
    const auto &decode_bboxes = all_decode_bboxes[b];
    std::map<int, VecInt> indices;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        LOG(FATAL) << "Could not find confidence predictions for label " << c;
      }
      const auto &scores = conf_scores.find(c)->second;
      int label = share_location_ ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        LOG(FATAL) << "Could not find confidence predictions for label "
                   << label;
      }
      const auto &bboxes = decode_bboxes.find(label)->second;
      ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_,
                   top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (const auto &it : indices) {
        int label = it.first;
        if (conf_scores.find(label) == conf_scores.end()) {
          LOG(FATAL) << "Could not find confidence predictions for" << label;
        }
        const auto &scores = conf_scores.find(label)->second;
        for (const auto &idx : it.second) {
          CHECK_LT(idx, scores.size());
          score_index_pairs.emplace_back(scores[idx],
                                         std::make_pair(label, idx));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<std::pair<int, int>>);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      std::map<int, VecInt> new_indices;
      for (const auto &score_index_pair : score_index_pairs) {
        int label = score_index_pair.second.first;
        int idx = score_index_pair.second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  Gboxes->clear();
  for (int b = 0; b < batch_; ++b) {
    VecBoxF boxes;
    const auto &conf_scores = all_conf_scores[b];
    const auto &decode_bboxes = all_decode_bboxes[b];
    for (const auto &it : all_indices[b]) {
      int label = it.first;
      if (conf_scores.find(label) == conf_scores.end()) {
        LOG(FATAL) << "Could not find confidence predictions for " << label;
      }
      const auto &scores = conf_scores.find(label)->second;
      int loc_label = share_location_ ? -1 : label;
      if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
        LOG(FATAL) << "Could not find confidence predictions for " << loc_label;
      }
      const auto &bboxes = decode_bboxes.find(loc_label)->second;
      for (const auto &idx : it.second) {
        float score = scores[idx];
        if (score > threshold_) {
          BoxF clip_bbox;
          Boxes::Clip(bboxes[idx], &clip_bbox, 0.f, 1.f);
          clip_bbox.score = score;
          clip_bbox.label = label;
          boxes.push_back(clip_bbox);
        }
      }
    }
    Gboxes->push_back(boxes);
  }
}

void DetectSSD::GetLocPredictions(const float *loc_data, int num,
                                  int num_preds_per_class, int num_loc_classes,
                                  bool share_location,
                                  VecLabelBBox *loc_preds) {
  loc_preds->resize(num);
  for (auto &label_bbox : *loc_preds) {
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        int label = share_location ? -1 : c;
        if (label_bbox.find(label) == label_bbox.end()) {
          label_bbox[label].resize(num_preds_per_class);
        }
        label_bbox[label][p].xmin = loc_data[start_idx + c * 4];
        label_bbox[label][p].ymin = loc_data[start_idx + c * 4 + 1];
        label_bbox[label][p].xmax = loc_data[start_idx + c * 4 + 2];
        label_bbox[label][p].ymax = loc_data[start_idx + c * 4 + 3];
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4;
  }
}

void DetectSSD::GetConfidenceScores(
    const float *conf_data, int num, int num_preds_per_class, int num_classes,
    std::vector<std::map<int, VecFloat>> *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (auto &label_scores : *conf_preds) {
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        label_scores[c].push_back(conf_data[start_idx + c]);
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}

void DetectSSD::GetPriorBBoxes(const float *prior_data, int num_priors,
                               VecBoxF *prior_bboxes,
                               std::vector<VecFloat> *prior_variances) {
  prior_bboxes->clear();
  prior_variances->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    BoxF bbox;
    bbox.xmin = prior_data[start_idx];
    bbox.ymin = prior_data[start_idx + 1];
    bbox.xmax = prior_data[start_idx + 2];
    bbox.ymax = prior_data[start_idx + 3];
    prior_bboxes->push_back(bbox);
  }

  for (int i = 0; i < num_priors; ++i) {
    int start_idx = (num_priors + i) * 4;
    VecFloat var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

void DetectSSD::DecodeBBoxesAll(const VecLabelBBox &all_loc_preds,
                                const VecBoxF &prior_bboxes,
                                const std::vector<VecFloat> &prior_variances,
                                int num, bool share_location,
                                int num_loc_classes, int background_label_id,
                                VecLabelBBox *all_decode_bboxes) {
  CHECK_EQ(all_loc_preds.size(), num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  for (int i = 0; i < num; ++i) {
    auto &decode_bboxes = all_decode_bboxes->at(i);
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        LOG(FATAL) << "Could not find location predictions for label " << label;
      }
      const auto &label_loc_preds = all_loc_preds[i].find(label)->second;
      DecodeBBoxes(prior_bboxes, prior_variances, label_loc_preds,
                   &(decode_bboxes[label]));
    }
  }
}

void DetectSSD::DecodeBBoxes(const VecBoxF &prior_bboxes,
                             const std::vector<VecFloat> &prior_variances,
                             const VecBoxF &bboxes, VecBoxF *decode_bboxes) {
  CHECK_EQ(prior_bboxes.size(), prior_variances.size());
  CHECK_EQ(prior_bboxes.size(), bboxes.size());
  size_t num_bboxes = prior_bboxes.size();
  if (num_bboxes >= 1) {
    CHECK_EQ(prior_variances[0].size(), 4);
  }
  decode_bboxes->clear();
  for (int i = 0; i < num_bboxes; ++i) {
    BoxF decode_bbox;
    DecodeBBox(prior_bboxes[i], prior_variances[i], bboxes[i], &decode_bbox);
    decode_bboxes->push_back(decode_bbox);
  }
}

void DetectSSD::DecodeBBox(const BoxF &prior_bbox,
                           const VecFloat &prior_variance, const BoxF &bbox,
                           BoxF *decode_bbox) {
  float prior_width = prior_bbox.xmax - prior_bbox.xmin;
  CHECK_GT(prior_width, 0);
  float prior_height = prior_bbox.ymax - prior_bbox.ymin;
  CHECK_GT(prior_height, 0);
  float prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.f;
  float prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.f;

  float decode_bbox_center_x, decode_bbox_center_y;
  float decode_bbox_width, decode_bbox_height;

  // variance is encoded in bbox, we need to scale the offset accordingly.
  decode_bbox_center_x =
      prior_variance[0] * bbox.xmin * prior_width + prior_center_x;
  decode_bbox_center_y =
      prior_variance[1] * bbox.ymin * prior_height + prior_center_y;
  decode_bbox_width = std::exp(prior_variance[2] * bbox.xmax) * prior_width;
  decode_bbox_height = std::exp(prior_variance[3] * bbox.ymax) * prior_height;

  decode_bbox->xmin = decode_bbox_center_x - decode_bbox_width / 2.f;
  decode_bbox->ymin = decode_bbox_center_y - decode_bbox_height / 2.f;
  decode_bbox->xmax = decode_bbox_center_x + decode_bbox_width / 2.f;
  decode_bbox->ymax = decode_bbox_center_y + decode_bbox_height / 2.f;
}

}  // namespace Shadow
