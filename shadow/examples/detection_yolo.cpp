#include "detection_yolo.hpp"

namespace Shadow {

void DetectionYOLO::Setup(const std::string &model_file) {
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

  out_str_ = net_.out_blob();

  const auto &data_shape = net_.GetBlobShapeByName<float>(in_str_);
  CHECK_EQ(data_shape.size(), 4);

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  threshold_ = 0.6;
  nms_threshold_ = 0.3;
  num_classes_ = net_.get_single_argument<int>("num_classes", 80);

  // For yolo v2
  biases_ = {{0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f, 5.47434f,
              7.88282f, 3.52778f, 9.77052f, 9.16828f}};
  num_km_ = 5;
  version_ = 2;

  // For yolo v3
  // biases_ = {{81, 82, 135, 169, 344, 319}, {10, 14, 23, 27, 37, 58}};
  // num_km_ = 3;
  // version_ = 3;

  CHECK_EQ(out_str_.size(), biases_.size());
}

void DetectionYOLO::Predict(const JImage &im_src, const RectF &roi,
                            VecBoxF *boxes, std::vector<VecPointF> *Gpoints) {
  ConvertData(im_src, in_data_.data(), roi, in_c_, in_h_, in_w_, 0);

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
void DetectionYOLO::Predict(const cv::Mat &im_mat, const RectF &roi,
                            VecBoxF *boxes, std::vector<VecPointF> *Gpoints) {
  ConvertData(im_mat, in_data_.data(), roi, in_c_, in_h_, in_w_, 0);

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

void DetectionYOLO::Process(const VecFloat &in_data,
                            std::vector<VecBoxF> *Gboxes) {
  std::map<std::string, float *> data_map;
  data_map[in_str_] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  Gboxes->clear();
  for (int b = 0; b < batch_; ++b) {
    VecBoxF all_boxes;
    for (int n = 0; n < out_str_.size(); ++n) {
      const auto &out_shape = net_.GetBlobShapeByName<float>(out_str_[n]);
      const auto *out_data = net_.GetBlobDataByName<float>(out_str_[n]);

      int out_num = 1;
      for (int d = 1; d < out_shape.size(); ++d) {
        out_num *= out_shape[d];
      }

      VecBoxF boxes;
      ConvertDetections(const_cast<float *>(out_data) + b * out_num,
                        biases_[n].data(), out_shape[1], out_shape[2], &boxes);
      all_boxes.insert(all_boxes.begin(), boxes.begin(), boxes.end());
    }
    Gboxes->push_back(Boxes::NMS(all_boxes, nms_threshold_));
  }
}

inline float activate(float x) { return 1.f / (1 + std::exp(-x)); }

inline void softmax(float *scores, int n) {
  float largest = -FLT_MAX;
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    if (scores[i] > largest) largest = scores[i];
  }
  for (int i = 0; i < n; ++i) {
    float e = std::exp(scores[i] - largest);
    sum += e;
    scores[i] = e;
  }
  for (int i = 0; i < n; ++i) {
    scores[i] /= sum;
  }
}

inline void ActivateSoftmax(int version, float *data, int classes, int num_km,
                            int out_h, int out_w) {
  for (int n = 0; n < out_h * out_w * num_km; ++n) {
    int offset = n * (4 + 1 + classes);
    data[offset + 0] = activate(data[offset + 0]);
    data[offset + 1] = activate(data[offset + 1]);
    data[offset + 4] = activate(data[offset + 4]);
    if (version == 2) {
      softmax(data + offset + 5, classes);
    } else if (version == 3) {
      for (int c = 0; c < classes; ++c) {
        data[offset + 5 + c] = activate(data[offset + 5 + c]);
      }
    } else {
      LOG(FATAL) << "Unsupported yolo version " << version;
    }
  }
}

inline void ConvertBoxes(int version, const float *data, const float *biases,
                         int classes, int num_km, int in_h, int in_w, int out_h,
                         int out_w, float threshold, VecBoxF *boxes) {
  boxes->clear();
  for (int n = 0; n < out_h * out_w * num_km; ++n) {
    int s = n / num_km, k = n % num_km;
    int row = s / out_w, col = s % out_w;
    int offset = n * (4 + 1 + classes);

    float max_score = -1;
    int max_index = -1;
    float scale = data[offset + 4];
    for (int c = 0; c < classes; ++c) {
      float score = scale * data[offset + 5 + c];
      if (score > max_score) {
        max_score = score;
        max_index = c;
      }
    }
    if (max_score > threshold) {
      float x = (data[offset + 0] + col) / out_w;
      float y = (data[offset + 1] + row) / out_h;
      float w = std::exp(data[offset + 2]) * biases[2 * k] / out_w;
      float h = std::exp(data[offset + 3]) * biases[2 * k + 1] / out_h;

      if (version == 3) {
        w = w * out_w / in_w;
        h = h * out_h / in_h;
      }

      BoxF box;
      box.xmin = Util::constrain(0.f, 1.f, x - w / 2);
      box.ymin = Util::constrain(0.f, 1.f, y - h / 2);
      box.xmax = Util::constrain(0.f, 1.f, x + w / 2);
      box.ymax = Util::constrain(0.f, 1.f, y + h / 2);
      box.score = max_score;
      box.label = max_index;
      boxes->push_back(box);
    }
  }
}

void DetectionYOLO::ConvertDetections(float *data, const float *biases,
                                      int out_h, int out_w, VecBoxF *boxes) {
  ActivateSoftmax(version_, data, num_classes_, num_km_, out_h, out_w);
  ConvertBoxes(version_, data, biases, num_classes_, num_km_, in_h_, in_w_,
               out_h, out_w, threshold_, boxes);
}

}  // namespace Shadow
