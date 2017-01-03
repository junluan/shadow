#ifndef SHADOW_YOLO_HPP
#define SHADOW_YOLO_HPP

#include "shadow/network.hpp"
#include "shadow/util/boxes.hpp"
#include "shadow/util/jimage.hpp"
#include "shadow/util/util.hpp"

class YOLO {
 public:
  void Setup(const std::string &model_file, int batch = 1);

  void Predict(const JImage &image, const VecRectF &rois,
               std::vector<VecBoxF> *Bboxes);
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<VecBoxF> *Bboxes);
#endif

  void Release();

 private:
  void Process(const float *data, std::vector<VecBoxF> *Bboxes);

  void ConvertDetections(float *data, float *biases, int classes, int num_km,
                         int side, float threshold, VecBoxF *boxes);

  Network net_;
  VecFloat in_data_, out_data_, biases_;
  int batch_, in_num_, in_c_, in_h_, in_w_, out_num_, out_hw_;
  int num_classes_, num_km_;
  float threshold_;
  JImage im_ini_, im_res_;
};

typedef YOLO METHOD;

#endif  // SHADOW_YOLO_HPP
