#include "detection_faster_rcnn.hpp"

namespace Shadow {

inline bool SortBoxesDescend(const BoxF &box_a, const BoxF &box_b) {
  return box_a.score > box_b.score;
}

inline VecBoxF NMS(const VecBoxF &boxes, float threshold) {
  auto all_boxes = boxes;
  std::stable_sort(all_boxes.begin(), all_boxes.end(), SortBoxesDescend);
  for (int i = 0; i < all_boxes.size(); ++i) {
    auto &box_i = all_boxes[i];
    if (box_i.label == -1) continue;
    for (int j = i + 1; j < all_boxes.size(); ++j) {
      auto &box_j = all_boxes[j];
      if (box_j.label == -1 || box_i.label != box_j.label) continue;
      if (Boxes::IoU(box_i, box_j) > threshold) {
        box_j.label = -1;
        continue;
      }
    }
  }
  VecBoxF out_boxes;
  for (const auto &box : all_boxes) {
    if (box.label != -1) {
      out_boxes.push_back(box);
    }
  }
  all_boxes.clear();
  return out_boxes;
}

void DetectionFasterRCNN::Setup(const VecString &model_files,
                                const VecInt &in_shape) {
  net_.Setup();

  net_.LoadModel(model_files[0]);

  in_shape_ = net_.GetBlobByName<float>("data")->shape();
  CHECK_EQ(in_shape_[0], 1);

  im_info_.resize(3, 0);
  num_classes_ = net_.num_class()[0];
  max_side_ = 1000, min_side_ = {600};
  nms_threshold_ = 0.3;
  confidence_threshold_ = 0.05;
}

void DetectionFasterRCNN::Predict(
    const JImage &im_src, const VecRectF &rois, std::vector<VecBoxF> *Gboxes,
    std::vector<std::vector<VecPointF>> *Gpoints) {
  Gboxes->clear();
  for (const auto &roi : rois) {
    float crop_h = roi.h <= 1 ? roi.h * im_src.h_ : roi.h;
    float crop_w = roi.w <= 1 ? roi.w * im_src.w_ : roi.w;
    CalculateScales(crop_h, crop_w, max_side_, min_side_, &scales_);

    auto scale_h = static_cast<int>(std::ceil(crop_h * scales_[0]));
    auto scale_w = static_cast<int>(std::ceil(crop_w * scales_[0]));
    in_shape_[2] = scale_h, in_shape_[3] = scale_w;
    im_info_[0] = scale_h, im_info_[1] = scale_w, im_info_[2] = scales_[0];
    in_data_.resize(1 * 3 * scale_h * scale_w);
    ConvertData(im_src, in_data_.data(), roi, 3, scale_h, scale_w, 1);

    VecBoxF boxes;
    Process(in_data_, in_shape_, im_info_, crop_h, crop_w, &boxes);

    Gboxes->push_back(boxes);
  }
}

#if defined(USE_OpenCV)
void DetectionFasterRCNN::Predict(
    const cv::Mat &im_mat, const VecRectF &rois, std::vector<VecBoxF> *Gboxes,
    std::vector<std::vector<VecPointF>> *Gpoints) {
  im_ini_.FromMat(im_mat, true);
  Predict(im_ini_, rois, Gboxes, Gpoints);
}
#endif

void DetectionFasterRCNN::Release() { net_.Release(); }

void DetectionFasterRCNN::Process(const VecFloat &in_data,
                                  const VecInt &in_shape,
                                  const VecFloat &im_info, float height,
                                  float width, VecBoxF *boxes) {
  std::map<std::string, VecInt> shape_map;
  std::map<std::string, float *> data_map;
  shape_map["data"] = in_shape;
  data_map["data"] = const_cast<float *>(in_data.data());
  data_map["im_info"] = const_cast<float *>(im_info.data());

  net_.Reshape(shape_map);
  net_.Forward(data_map);

  const auto *roi_blob = net_.GetBlobByName<float>("rois");
  const auto *roi_data = net_.GetBlobDataByName<float>("rois");
  const auto *score_data = net_.GetBlobDataByName<float>("cls_prob");
  const auto *delta_data = net_.GetBlobDataByName<float>("bbox_pred");

  boxes->clear();
  int num_rois = roi_blob->shape(0);
  for (int n = 0; n < num_rois; ++n) {
    int label = 0, score_offset = n * num_classes_;
    float max_score = 0.f;
    for (int c = 0; c < num_classes_; ++c) {
      float score = score_data[score_offset + c];
      if (score > max_score) {
        label = c;
        max_score = score;
      }
    }
    if (label == 0 || max_score < confidence_threshold_) continue;

    float xmin = roi_data[n * 5 + 1] / im_info[2];
    float ymin = roi_data[n * 5 + 2] / im_info[2];
    float xmax = roi_data[n * 5 + 3] / im_info[2];
    float ymax = roi_data[n * 5 + 4] / im_info[2];

    float pb_w = xmax - xmin + 1;
    float pb_h = ymax - ymin + 1;
    float cx = xmin + pb_w * 0.5f;
    float cy = ymin + pb_h * 0.5f;

    float dx = delta_data[label * 4 + 0];
    float dy = delta_data[label * 4 + 1];
    float dw = delta_data[label * 4 + 2];
    float dh = delta_data[label * 4 + 3];

    float pred_cx = cx + pb_w * dx;
    float pred_cy = cy + pb_h * dy;
    float pred_w = pb_w * std::exp(dw);
    float pred_h = pb_h * std::exp(dh);

    BoxF box;
    box.label = label;
    box.score = max_score;

    box.xmin = pred_cx - pred_w * 0.5f;
    box.ymin = pred_cy - pred_h * 0.5f;
    box.xmax = pred_cx + pred_w * 0.5f;
    box.ymax = pred_cy + pred_h * 0.5f;

    box.xmin = std::max(std::min(box.xmin, width - 1), 0.f);
    box.ymin = std::max(std::min(box.ymin, height - 1), 0.f);
    box.xmax = std::max(std::min(box.xmax, width - 1), 0.f);
    box.ymax = std::max(std::min(box.ymax, height - 1), 0.f);

    boxes->push_back(box);
  }
  *boxes = NMS(*boxes, nms_threshold_);
}

void DetectionFasterRCNN::CalculateScales(float height, float width,
                                          float max_side,
                                          const VecFloat &min_side,
                                          VecFloat *scales) {
  scales->clear();
  float pr_min = std::min(height, width), pr_max = std::max(height, width);
  for (const auto side : min_side) {
    float scale = side / pr_min;
    if (scale * pr_max > max_side) {
      scale = max_side / pr_max;
    }
    scales->push_back(scale);
  }
}

}  // namespace Shadow
