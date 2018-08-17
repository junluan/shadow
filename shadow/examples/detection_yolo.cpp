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

  const auto &out_blob = net_.out_blob();
  CHECK_EQ(out_blob.size(), 1);
  out_str_ = out_blob[0];

  auto data_shape = net_.GetBlobByName<float>(in_str_)->shape();
  CHECK_EQ(data_shape.size(), 4);

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  biases_ = VecFloat{1.08f,  1.19f, 3.42f, 4.41f,  6.63f,
                     11.38f, 9.42f, 5.11f, 16.62f, 10.52f};
  threshold_ = 0.6;
  num_classes_ = net_.get_single_argument<int>("num_classes", 20);
  num_km_ = 5;
}

void DetectionYOLO::Predict(const JImage &im_src, const VecRectF &rois,
                            std::vector<VecBoxF> *Gboxes,
                            std::vector<std::vector<VecPointF>> *Gpoints) {
  CHECK_LE(rois.size(), batch_);
  for (int b = 0; b < rois.size(); ++b) {
    ConvertData(im_src, in_data_.data() + b * in_num_, rois[b], in_c_, in_h_,
                in_w_, 0);
  }

  Process(in_data_, Gboxes);

  CHECK_EQ(Gboxes->size(), rois.size());
  for (int b = 0; b < Gboxes->size(); ++b) {
    float height = rois[b].h, width = rois[b].w;
    for (auto &box : Gboxes->at(b)) {
      box.xmin *= width;
      box.xmax *= width;
      box.ymin *= height;
      box.ymax *= height;
    }
  }
}

#if defined(USE_OpenCV)
void DetectionYOLO::Predict(const cv::Mat &im_mat, const VecRectF &rois,
                            std::vector<VecBoxF> *Gboxes,
                            std::vector<std::vector<VecPointF>> *Gpoints) {
  CHECK_LE(rois.size(), batch_);
  for (int b = 0; b < rois.size(); ++b) {
    ConvertData(im_mat, in_data_.data() + b * in_num_, rois[b], in_c_, in_h_,
                in_w_, 0);
  }

  Process(in_data_, Gboxes);

  CHECK_EQ(Gboxes->size(), rois.size());
  for (int b = 0; b < Gboxes->size(); ++b) {
    float height = rois[b].h, width = rois[b].w;
    for (auto &box : Gboxes->at(b)) {
      box.xmin *= width;
      box.xmax *= width;
      box.ymin *= height;
      box.ymax *= height;
    }
  }
}
#endif

void DetectionYOLO::Release() { net_.Release(); }

void DetectionYOLO::Process(const VecFloat &in_data,
                            std::vector<VecBoxF> *Gboxes) {
  std::map<std::string, float *> data_map;
  data_map[in_str_] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  auto *out_blob = net_.GetBlobByName<float>(out_str_);
  auto *out_data = const_cast<float *>(out_blob->cpu_data());

  out_num_ = out_blob->num();
  out_hw_ = out_blob->shape(2);

  Gboxes->clear();
  for (int b = 0; b < batch_; ++b) {
    VecBoxF boxes;
    ConvertDetections(out_data + b * out_num_, biases_.data(), num_classes_,
                      num_km_, out_hw_, threshold_, &boxes);
    Gboxes->push_back(boxes);
  }
}

inline float logistic_activate(float x) { return 1.f / (1 + std::exp(-x)); }

inline void Softmax(float *scores, int n) {
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

inline void ActivateSoftmax(float *data, int classes, int num_km, int side) {
  int feature_size = 4 + classes + 1, num_box = side * side * num_km;
  for (int i = 0; i < num_box; ++i) {
    int index = i * feature_size;
    data[index + 4] = logistic_activate(data[index + 4]);
    Softmax(data + index + 5, classes);
  }
}

inline void ConvertRegionBoxes(const float *data, const float *biases,
                               int classes, int num_km, int side,
                               float threshold, VecBoxF *boxes) {
  boxes->clear();
  for (int i = 0; i < side * side; ++i) {
    int row = i / side, col = i % side;
    for (int n = 0; n < num_km; ++n) {
      int index = i * num_km + n;
      int p_index = index * (classes + 5) + 4;
      float scale = data[p_index];
      int box_index = index * (classes + 5);

      float x, y, w, h;
      x = (logistic_activate(data[box_index + 0]) + col) / side;
      y = (logistic_activate(data[box_index + 1]) + row) / side;
      w = std::exp(data[box_index + 2]) * biases[2 * n] / side;
      h = std::exp(data[box_index + 3]) * biases[2 * n + 1] / side;

      x = x - w / 2;
      y = y - h / 2;

      BoxF box;
      box.xmin = Util::constrain(0.f, 1.f, x);
      box.ymin = Util::constrain(0.f, 1.f, y);
      box.xmax = Util::constrain(0.f, 1.f, x + w);
      box.ymax = Util::constrain(0.f, 1.f, y + h);

      int class_index = index * (classes + 5) + 5;
      float max_score = 0;
      int max_index = -1;
      for (int j = 0; j < classes; ++j) {
        float score = scale * data[class_index + j];
        if (score > threshold && score > max_score) {
          max_score = score;
          max_index = j;
        }
      }
      box.score = max_score;
      box.label = max_index;
      boxes->push_back(box);
    }
  }
}

void DetectionYOLO::ConvertDetections(float *data, float *biases, int classes,
                                      int num_km, int side, float threshold,
                                      VecBoxF *boxes) {
  ActivateSoftmax(data, classes, num_km, side);
  ConvertRegionBoxes(data, biases, classes, num_km, side, threshold, boxes);
}

}  // namespace Shadow
