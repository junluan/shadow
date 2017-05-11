#include "yolo.hpp"
#include "util/jimage_proc.hpp"

namespace Shadow {

inline void ConvertData(const JImage &im_src, float *data, int flag = 1) {
  CHECK_NOTNULL(im_src.data());
  CHECK_NOTNULL(data);

  int h = im_src.h_, w = im_src.w_, spatial_dim = h * w;
  const auto &order = im_src.order();
  const unsigned char *data_src = im_src.data();

  float *data_r, *data_g, *data_b;
  if (flag == 0) {
    // Convert to RRRGGGBBB
    data_r = data;
    data_g = data + spatial_dim;
    data_b = data + (spatial_dim << 1);
  } else {
    // Convert to BBBGGGRRR
    data_r = data + (spatial_dim << 1);
    data_g = data + spatial_dim;
    data_b = data;
  }

  if (order == kRGB) {
    for (int i = 0; i < spatial_dim; ++i) {
      *(data_r++) = *data_src;
      *(data_g++) = *(data_src + 1);
      *(data_b++) = *(data_src + 2);
      data_src += 3;
    }
  } else if (order == kBGR) {
    for (int i = 0; i < spatial_dim; ++i) {
      *(data_r++) = *(data_src + 2);
      *(data_g++) = *(data_src + 1);
      *(data_b++) = *data_src;
      data_src += 3;
    }
  } else {
    LOG(FATAL) << "Unsupported format to get batch data!";
  }
}

void YOLO::Setup(const std::string &model_file, int classes, int batch) {
  net_.Setup();

  net_.LoadModel(model_file, batch);

  batch_ = net_.in_shape()[0];
  in_c_ = net_.in_shape()[1];
  in_h_ = net_.in_shape()[2];
  in_w_ = net_.in_shape()[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  out_num_ = net_.GetBlobByName("out_blob")->num();
  out_hw_ = net_.GetBlobByName("out_blob")->shape(2);

  in_data_.resize(batch_ * in_num_);
  out_data_.resize(batch_ * out_num_);

  biases_ = VecFloat{1.08f,  1.19f, 3.42f, 4.41f,  6.63f,
                     11.38f, 9.42f, 5.11f, 16.62f, 10.52f};
  threshold_ = 0.6;
  num_classes_ = classes;
  num_km_ = 5;
}

void YOLO::Predict(const JImage &image, const VecRectF &rois,
                   std::vector<VecBoxF> *Bboxes) {
  CHECK_LE(rois.size(), batch_);
  for (int b = 0; b < rois.size(); ++b) {
    JImageProc::CropResize(image, &im_res_, rois[b], in_h_, in_w_);
    ConvertData(im_res_, in_data_.data() + b * in_num_, 0);
  }

  Process(in_data_.data(), Bboxes);

  CHECK_EQ(Bboxes->size(), rois.size());
  for (int b = 0; b < Bboxes->size(); ++b) {
    float height = rois[b].h, width = rois[b].w;
    VecBoxF &boxes = Bboxes->at(b);
    for (auto &box : boxes) {
      box.xmin *= width;
      box.xmax *= width;
      box.ymin *= height;
      box.ymax *= height;
    }
  }
}

#if defined(USE_OpenCV)
void YOLO::Predict(const cv::Mat &im_mat, const VecRectF &rois,
                   std::vector<VecBoxF> *Bboxes) {
  im_ini_.FromMat(im_mat, true);
  Predict(im_ini_, rois, Bboxes);
}
#endif

void YOLO::Release() {
  net_.Release();

  in_data_.clear(), out_data_.clear(), biases_.clear();
}

void YOLO::Process(const float *data, std::vector<VecBoxF> *Bboxes) {
  net_.Forward(data);

  memcpy(out_data_.data(), net_.GetBlobDataByName("out_blob"),
         out_data_.size() * sizeof(float));

  Bboxes->clear();
  for (int b = 0; b < batch_; ++b) {
    VecBoxF boxes;
    ConvertDetections(out_data_.data() + b * out_num_, biases_.data(),
                      num_classes_, num_km_, out_hw_, threshold_, &boxes);
    Bboxes->push_back(boxes);
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

void YOLO::ConvertDetections(float *data, float *biases, int classes,
                             int num_km, int side, float threshold,
                             VecBoxF *boxes) {
  ActivateSoftmax(data, classes, num_km, side);
  ConvertRegionBoxes(data, biases, classes, num_km, side, threshold, boxes);
}

}  // namespace Shadow
