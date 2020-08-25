#include "detect_faster_rcnn.hpp"

#include "util/io.hpp"

namespace Shadow {

void DetectFasterRCNN::Setup(const std::string& model_file) {
#if defined(USE_Protobuf)
  shadow::MetaNetParam meta_net_param;
  CHECK(IO::ReadProtoFromBinaryFile(model_file, &meta_net_param))
      << "Error when loading proto binary file: " << model_file;

  ArgumentHelper arguments;
  arguments.AddSingleArgument<std::string>("backend_type", "Native");

  net_.LoadXModel(meta_net_param.network(0), arguments);

#else
  LOG(FATAL) << "Unsupported load binary model, recompiled with USE_Protobuf";
#endif

  const auto& in_blob = net_.in_blob();
  CHECK_EQ(in_blob.size(), 2);
  in_str_ = in_blob[0];
  im_info_str_ = in_blob[1];

  const auto& out_blob = net_.out_blob();
  CHECK_EQ(out_blob.size(), 3);
  rois_str_ = out_blob[0];
  cls_prob_str_ = out_blob[1];
  bbox_pred_str_ = out_blob[2];

  in_shape_ = net_.GetBlobShapeByName<float>(in_str_);
  CHECK_EQ(in_shape_[0], 1);

  im_info_.resize(3, 0);
  num_classes_ = net_.get_single_argument<int>("num_classes", 21);
  max_side_ = 1000, min_side_ = {600};
  threshold_ = net_.get_single_argument<float>("threshold", 0.6);
  nms_threshold_ = net_.get_single_argument<float>("nms_threshold", 0.3);
  is_bgr_ = net_.get_single_argument<bool>("is_bgr", true);
  class_agnostic_ = net_.get_single_argument<bool>("class_agnostic", false);
}

void DetectFasterRCNN::Predict(const cv::Mat& im_mat, const RectF& roi,
                               VecBoxF* boxes,
                               std::vector<VecPointF>* Gpoints) {
  float crop_h = roi.h <= 1 ? roi.h * im_mat.rows : roi.h;
  float crop_w = roi.w <= 1 ? roi.w * im_mat.cols : roi.w;
  CalculateScales(crop_h, crop_w, max_side_, min_side_, &scales_);

  auto scale_h = static_cast<int>(crop_h * scales_[0]);
  auto scale_w = static_cast<int>(crop_w * scales_[0]);
  in_shape_[2] = scale_h, in_shape_[3] = scale_w;
  im_info_[0] = scale_h, im_info_[1] = scale_w, im_info_[2] = scales_[0];
  in_data_.resize(1 * 3 * scale_h * scale_w);
  if (is_bgr_) {
    ConvertData(im_mat, in_data_.data(), roi, 3, scale_h, scale_w, 1);
  } else {
    ConvertData(im_mat, in_data_.data(), roi, 3, scale_h, scale_w, 0);
  }

  Process(in_data_, in_shape_, im_info_, crop_h, crop_w, boxes);
}

void DetectFasterRCNN::Process(const VecFloat& in_data, const VecInt& in_shape,
                               const VecFloat& im_info, float height,
                               float width, VecBoxF* boxes) {
  std::map<std::string, void*> data_map;
  std::map<std::string, VecInt> shape_map;
  data_map[in_str_] = const_cast<float*>(in_data.data());
  data_map[im_info_str_] = const_cast<float*>(im_info.data());
  shape_map[in_str_] = in_shape;

  net_.Forward(data_map, shape_map);

  const auto& roi_shape = net_.GetBlobShapeByName<float>(rois_str_);

  const auto* roi_data = net_.GetBlobDataByName<float>(rois_str_);
  const auto* score_data = net_.GetBlobDataByName<float>(cls_prob_str_);
  const auto* delta_data = net_.GetBlobDataByName<float>(bbox_pred_str_);

  boxes->clear();
  int num_rois = roi_shape[0];
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
    if (label == 0 || max_score < threshold_) continue;

    float pb_xmin = roi_data[n * 5 + 1] / im_info[2];
    float pb_ymin = roi_data[n * 5 + 2] / im_info[2];
    float pb_xmax = roi_data[n * 5 + 3] / im_info[2];
    float pb_ymax = roi_data[n * 5 + 4] / im_info[2];

    float pb_w = pb_xmax - pb_xmin + 1;
    float pb_h = pb_ymax - pb_ymin + 1;
    float pb_cx = pb_xmin + (pb_w - 1) * 0.5f;
    float pb_cy = pb_ymin + (pb_h - 1) * 0.5f;

    int delta_offset;
    if (class_agnostic_) {
      delta_offset = (n * 2 + 1) * 4;
    } else {
      delta_offset = (n * num_classes_ + label) * 4;
    }
    float dx = delta_data[delta_offset + 0];
    float dy = delta_data[delta_offset + 1];
    float dw = delta_data[delta_offset + 2];
    float dh = delta_data[delta_offset + 3];

    float pred_cx = pb_cx + pb_w * dx;
    float pred_cy = pb_cy + pb_h * dy;
    float pred_w = pb_w * std::exp(dw);
    float pred_h = pb_h * std::exp(dh);

    BoxF box;
    box.label = label;
    box.score = max_score;

    box.xmin = pred_cx - (pred_w - 1) * 0.5f;
    box.ymin = pred_cy - (pred_h - 1) * 0.5f;
    box.xmax = pred_cx + (pred_w - 1) * 0.5f;
    box.ymax = pred_cy + (pred_h - 1) * 0.5f;

    box.xmin = std::min(std::max(box.xmin, 0.f), width - 1);
    box.ymin = std::min(std::max(box.ymin, 0.f), height - 1);
    box.xmax = std::min(std::max(box.xmax, 0.f), width - 1);
    box.ymax = std::min(std::max(box.ymax, 0.f), height - 1);

    boxes->push_back(box);
  }
  *boxes = Boxes::NMS(*boxes, nms_threshold_);
}

void DetectFasterRCNN::CalculateScales(float height, float width,
                                       float max_side, const VecFloat& min_side,
                                       VecFloat* scales) {
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
