#ifndef SHADOW_YOLO_HPP
#define SHADOW_YOLO_HPP

#include "shadow/network.hpp"
#include "shadow/util/boxes.hpp"
#include "shadow/util/jimage.hpp"
#include "shadow/util/util.hpp"

class Yolo {
 public:
  Yolo(std::string cfg_file, std::string weight_file, float threshold);
  ~Yolo();

  void Setup(int batch = 1, VecRectF *rois = nullptr);
  void Test(std::string image_file);
  void BatchTest(std::string list_file, bool image_write = false);
#if defined(USE_OpenCV)
  void VideoTest(std::string video_file, bool video_show = false,
                 bool video_write = false);
  void Demo(int camera, bool video_write = false);
#endif
  void Release();

 private:
  void PredictYoloDetections(JImage *image, std::vector<VecBox> *Bboxes);
  void ConvertYoloDetections(float *predictions, int classes, int width,
                             int height, VecBox *boxes);
#if defined(USE_OpenCV)
  void CaptureTest(cv::VideoCapture capture, std::string window_name,
                   bool video_show, cv::VideoWriter writer, bool video_write);
  void DrawYoloDetections(const VecBox &boxes, cv::Mat *im_mat,
                          bool console_show = true);
#endif
  void PrintYoloDetections(const VecBox &boxes, int count, std::ofstream *file);

  std::string cfg_file_, weight_file_;
  float threshold_;
  Network net_;
  const Blob<float> *out_blob_;
  int batch_, in_num_, out_num_, class_num_;
  float *batch_data_, *predictions_;
  JImage *im_ini_, *im_res_;
  VecRectF rois_;
};

#endif  // SHADOW_YOLO_HPP
