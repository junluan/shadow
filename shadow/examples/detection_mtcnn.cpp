#include "detection_mtcnn.hpp"

namespace Shadow {

void DetectionMTCNN::Setup(const VecString &model_files, const VecInt &classes,
                           const VecInt &in_shape) {
  net_12_.Setup();

  net_12_.LoadModel(model_files[0], VecInt{1, 3, 360, 360});
  net_24_.LoadModel(model_files[1], VecInt{10});
  net_48_.LoadModel(model_files[2], VecInt{10});

  net_12_in_shape_ = net_12_.in_shape();

  net_24_in_shape_ = net_24_.in_shape();
  net_24_in_c_ = net_24_in_shape_[1];
  net_24_in_h_ = net_24_in_shape_[2];
  net_24_in_w_ = net_24_in_shape_[3];
  net_24_in_num_ = net_24_in_c_ * net_24_in_h_ * net_24_in_w_;

  net_48_in_shape_ = net_48_.in_shape();
  net_48_in_c_ = net_48_in_shape_[1];
  net_48_in_h_ = net_48_in_shape_[2];
  net_48_in_w_ = net_48_in_shape_[3];
  net_48_in_num_ = net_48_in_c_ * net_48_in_h_ * net_48_in_w_;

  factor_ = 0.7f, max_side_ = 360, min_side_ = 12;
  thresholds_ = {0.6f, 0.6f, 0.7f};
  nms_thresholds_ = {0.7f, 0.7f, 0.7f};
}

void DetectionMTCNN::Predict(const JImage &im_src, const VecRectF &rois,
                             std::vector<VecBoxF> *Bboxes) {
  Bboxes->clear();
  net_12_boxes_.clear(), net_24_boxes_.clear(), net_48_boxes_.clear();
  for (const auto &roi : rois) {
    float crop_h = roi.h <= 1 ? roi.h * im_src.h_ : roi.h;
    float crop_w = roi.w <= 1 ? roi.w * im_src.w_ : roi.w;
    CalculateScales(crop_h, crop_w, factor_, max_side_, min_side_, &scales_);

    for (auto scale : scales_) {
      auto scale_h = static_cast<int>(crop_h * scale);
      auto scale_w = static_cast<int>(crop_w * scale);
      net_12_in_shape_[2] = scale_w, net_12_in_shape_[3] = scale_h;
      net_12_in_data_.resize(1 * 3 * scale_w * scale_h);
      ConvertData(im_src, net_12_in_data_.data(), roi, 3, scale_h, scale_w, 1,
                  true);
      VecBoxF boxes;
      Process_net_12(net_12_in_data_.data(), net_12_in_shape_, crop_h, crop_w,
                     thresholds_[0], scale, &boxes);
      net_12_boxes_.insert(net_12_boxes_.end(), boxes.begin(), boxes.end());
    }
    net_12_boxes_ = Boxes::NMS(net_12_boxes_, nms_thresholds_[0]);
    if (net_12_boxes_.empty()) {
      Bboxes->push_back(net_12_boxes_);
      continue;
    }

    net_24_in_shape_[0] = static_cast<int>(net_12_boxes_.size());
    net_24_in_data_.resize(net_24_in_shape_[0] * net_24_in_num_);
    for (int n = 0; n < net_12_boxes_.size(); ++n) {
      const auto &net_12_box = net_12_boxes_[n];
      ConvertData(im_src, net_24_in_data_.data() + n * net_24_in_num_,
                  net_12_box.RectFloat(), net_24_in_c_, net_24_in_h_,
                  net_24_in_w_, 1, true);
    }
    Process_net_24(net_24_in_data_.data(), net_24_in_shape_, crop_h, crop_w,
                   thresholds_[1], net_12_boxes_, &net_24_boxes_);
    net_24_boxes_ = Boxes::NMS(net_24_boxes_, nms_thresholds_[1]);
    if (net_24_boxes_.empty()) {
      Bboxes->push_back(net_24_boxes_);
      continue;
    }

    net_48_in_shape_[0] = static_cast<int>(net_24_boxes_.size());
    net_48_in_data_.resize(net_48_in_shape_[0] * net_48_in_num_);
    for (int n = 0; n < net_24_boxes_.size(); ++n) {
      const auto &net_24_box = net_24_boxes_[n];
      ConvertData(im_src, net_48_in_data_.data() + n * net_48_in_num_,
                  net_24_box.RectFloat(), net_48_in_c_, net_48_in_h_,
                  net_48_in_w_, 1, true);
    }
    Process_net_48(net_48_in_data_.data(), net_48_in_shape_, crop_h, crop_w,
                   thresholds_[2], net_24_boxes_, &net_48_boxes_);
    net_48_boxes_ = Boxes::NMS(net_48_boxes_, nms_thresholds_[2]);
    Bboxes->push_back(net_48_boxes_);
  }
}

#if defined(USE_OpenCV)
void DetectionMTCNN::Predict(const cv::Mat &im_mat, const VecRectF &rois,
                             std::vector<VecBoxF> *Bboxes) {
  im_ini_.FromMat(im_mat, true);
  Predict(im_ini_, rois, Bboxes);
}
#endif

void DetectionMTCNN::Release() {
  net_12_.Release();
  net_24_.Release();
  net_48_.Release();
}

void DetectionMTCNN::Process_net_12(const float *data, const VecInt &in_shape,
                                    float height, float width, float threshold,
                                    float scale, VecBoxF *boxes) {
  net_12_.Reshape(in_shape);
  net_12_.Forward(data);

  const auto *loc_blob = net_12_.GetBlobByName<float>("conv4-2");
  const auto *conf_blob = net_12_.GetBlobByName<float>("prob1");
  const auto *loc_data = const_cast<BlobF *>(loc_blob)->cpu_data();
  const auto *conf_data = const_cast<BlobF *>(conf_blob)->cpu_data();

  int out_h = loc_blob->shape(2), out_w = loc_blob->shape(3);
  int out_spatial_dim = out_h * out_w;
  int out_side = std::max(out_h, out_w), in_side = 2 * out_side + 11;
  float stride = 0;
  if (out_side != 1) {
    stride = static_cast<float>(in_side - 12) / (out_side - 1);
  }

  boxes->clear();
  for (int h = 0; h < out_h; ++h) {
    for (int w = 0; w < out_w; ++w) {
      int offset = h * out_w + w;
      float conf = conf_data[out_spatial_dim + offset];
      if (conf > threshold) {
        float x_min = (stride * h + 0) / scale;
        float y_min = (stride * w + 0) / scale;
        float x_max = (stride * h + 11) / scale;
        float y_max = (stride * w + 11) / scale;

        float x_min_offset = loc_data[offset];
        float y_min_offset = loc_data[out_spatial_dim + offset];
        float x_max_offset = loc_data[out_spatial_dim * 2 + offset];
        float y_max_offset = loc_data[out_spatial_dim * 3 + offset];

        x_min += x_min_offset * 12.f / scale;
        y_min += y_min_offset * 12.f / scale;
        x_max += x_max_offset * 12.f / scale;
        y_max += y_max_offset * 12.f / scale;

        BoxF box(x_min, y_min, x_max, y_max);
        box = Rect2SquareWithConstrain(box, height, width);
        if (box.xmax > box.xmin && box.ymax > box.ymin) {
          box.score = conf;
          box.label = 1;
          boxes->push_back(box);
        }
      }
    }
  }
  *boxes = Boxes::NMS(*boxes, 0.5);
}

void DetectionMTCNN::Process_net_24(const float *data, const VecInt &in_shape,
                                    float height, float width, float threshold,
                                    const VecBoxF &net_12_boxes,
                                    VecBoxF *boxes) {
  net_24_.Reshape(in_shape);
  net_24_.Forward(data);

  const auto *loc_data = net_24_.GetBlobDataByName<float>("conv5-2");
  const auto *conf_data = net_24_.GetBlobDataByName<float>("prob1");

  boxes->clear();
  for (int b = 0; b < in_shape[0]; ++b) {
    int loc_offset = 4 * b, conf_offset = 2 * b;
    float conf = conf_data[conf_offset + 1];
    if (conf > threshold) {
      const auto &net_12_box = net_12_boxes[b];
      float x_min = net_12_box.xmin;
      float y_min = net_12_box.ymin;
      float x_max = net_12_box.xmax;
      float y_max = net_12_box.ymax;

      float x_min_offset = loc_data[loc_offset];
      float y_min_offset = loc_data[loc_offset + 1];
      float x_max_offset = loc_data[loc_offset + 2];
      float y_max_offset = loc_data[loc_offset + 3];

      float rect_h = y_max - y_min, rect_w = x_max - x_min;
      x_min += x_min_offset * rect_w;
      y_min += y_min_offset * rect_h;
      x_max += x_max_offset * rect_w;
      y_max += y_max_offset * rect_h;

      BoxF box(x_min, y_min, x_max, y_max);
      box = Rect2SquareWithConstrain(box, height, width);
      if (box.xmax > box.xmin && box.ymax > box.ymin) {
        box.score = conf;
        box.label = 1;
        boxes->push_back(box);
      }
    }
  }
}

void DetectionMTCNN::Process_net_48(const float *data, const VecInt &in_shape,
                                    float height, float width, float threshold,
                                    const VecBoxF &net_24_boxes,
                                    VecBoxF *boxes) {
  net_48_.Reshape(in_shape);
  net_48_.Forward(data);

  const auto *loc_data = net_48_.GetBlobDataByName<float>("conv6-2");
  const auto *conf_data = net_48_.GetBlobDataByName<float>("prob1");

  boxes->clear();
  for (int b = 0; b < in_shape[0]; ++b) {
    int loc_offset = 4 * b, conf_offset = 2 * b;
    float conf = conf_data[conf_offset + 1];
    if (conf > threshold) {
      const auto &net_24_box = net_24_boxes[b];
      float x_min = net_24_box.xmin;
      float y_min = net_24_box.ymin;
      float x_max = net_24_box.xmax;
      float y_max = net_24_box.ymax;

      float x_min_offset = loc_data[loc_offset];
      float y_min_offset = loc_data[loc_offset + 1];
      float x_max_offset = loc_data[loc_offset + 2];
      float y_max_offset = loc_data[loc_offset + 3];

      float rect_h = y_max - y_min, rect_w = x_max - x_min;
      x_min += x_min_offset * rect_w;
      y_min += y_min_offset * rect_h;
      x_max += x_max_offset * rect_w;
      y_max += y_max_offset * rect_h;

      BoxF box(x_min, y_min, x_max, y_max);
      box = Rect2SquareWithConstrain(box, height, width);
      if (box.xmax > box.xmin && box.ymax > box.ymin) {
        box.score = conf;
        box.label = 1;
        boxes->push_back(box);
      }
    }
  }
}

void DetectionMTCNN::CalculateScales(float height, float width, float factor,
                                     float max_side, float min_side,
                                     VecFloat *scales) {
  scales->clear();
  float pr_scale = max_side / std::max(height, width);
  auto ini_h = height * pr_scale, ini_w = width * pr_scale;
  float pr_min = std::min(ini_h, ini_w);
  while (pr_min > min_side) {
    scales->push_back(pr_scale);
    pr_scale *= factor, ini_h *= factor, ini_w *= factor;
    pr_min = std::min(ini_h, ini_w);
  }
}

BoxF DetectionMTCNN::Rect2SquareWithConstrain(const BoxF &box, float height,
                                              float width) {
  float box_h = box.ymax - box.ymin, box_w = box.xmax - box.xmin;
  float box_l = std::max(box_h, box_w);
  float x_min = box.xmin + (box_w - box_l) / 2;
  float y_min = box.ymin + (box_h - box_l) / 2;
  float x_max = x_min + box_l;
  float y_max = y_min + box_l;
  x_min = std::max(0.f, x_min);
  y_min = std::max(0.f, y_min);
  x_max = std::min(width, x_max);
  y_max = std::min(height, y_max);
  return BoxF(x_min, y_min, x_max, y_max);
}

}  // namespace Shadow
