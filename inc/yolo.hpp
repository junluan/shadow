#ifndef SHADOW_YOLO_HPP
#define SHADOW_YOLO_HPP

#include "boxes.hpp"
#include "jimage.hpp"
#include "network.hpp"

#include <fstream>
#include <string>
#include <vector>

class Yolo {
public:
  Yolo(std::string cfgfile, std::string weightfile, float threshold);
  ~Yolo();

  void Setup(int batch = 1, VecBox *rois = nullptr);
  void Test(std::string imagefile);
  void BatchTest(std::string listfile, bool image_write = false);
#ifdef USE_OpenCV
  void VideoTest(std::string videofile, bool video_show = false,
                 bool video_write = false);
  void Demo(int camera, bool video_write = false);
#endif
  void Release();

private:
  std::string cfgfile_, weightfile_;
  float threshold_;
  Network net_;
  float *batch_data_, *predictions_;
  JImage *im_ini_, *im_crop_, *im_res_;
  VecBox rois_;

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
