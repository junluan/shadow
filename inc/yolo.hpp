#ifndef SHADOW_YOLO_HPP
#define SHADOW_YOLO_HPP

#include "boxes.hpp"
#include "jimage.hpp"
#include "network.hpp"

#include <string>
#include <vector>
#include <fstream>

class Yolo {
public:
  Yolo(std::string cfg_file, std::string weight_file, float threshold);
  ~Yolo();

  void Setup(int batch = 1, VecRectF *rois = nullptr);
  void Test(std::string image_file);
  void BatchTest(std::string list_file, bool image_write = false);
#ifdef USE_OpenCV
  void VideoTest(std::string video_file, bool video_show = false,
                 bool video_write = false);
  void Demo(int camera, bool video_write = false);
#endif
  void Release();

private:
  std::string cfg_file_, weight_file_;
  float threshold_;
  Network net_;
  float *batch_data_, *predictions_;
  JImage *im_ini_, *im_res_;
  VecRectF rois_;

  void PredictYoloDetections(JImage *image, std::vector<VecBox> *Bboxes);
  void ConvertYoloDetections(float *predictions, int classes, int num,
                             int square, int side, int width, int height,
                             VecBox *boxes);
#ifdef USE_OpenCV
  void CaptureTest(cv::VideoCapture capture, std::string window_name,
                   bool video_show, cv::VideoWriter writer, bool video_write);
  void DrawYoloDetections(const VecBox &boxes, cv::Mat *im_mat,
                          bool console_show = true);
#endif
  void PrintYoloDetections(const VecBox &boxes, int count, std::ofstream *file);
};

#endif // SHADOW_YOLO_HPP
