#ifndef SHADOW_YOLO_H
#define SHADOW_YOLO_H

#include "boxes.h"
#include "jimage.h"
#include "network.h"

#include <fstream>
#include <string>

class Yolo {
public:
  Yolo(std::string cfgfile, std::string weightfile, float threshold);
  ~Yolo();

  void Setup(Box *roi = nullptr);
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
  int class_num_, grid_size_, box_num_, sqrt_box_, out_num_;
  float *batch_data_, *predictions_;
  JImage *im_ini_, *im_crop_, *im_res_;
  Box *roi_;

  void PredictYoloDetections(std::vector<JImage *> &images,
                             std::vector<VecBox> &Bboxes);
  void ConvertYoloDetections(float *predictions, int classes, int num,
                             int square, int side, int width, int height,
                             VecBox &boxes);
#ifdef USE_OpenCV
  void CaptureTest(cv::VideoCapture capture, std::string window_name,
                   bool video_show, cv::VideoWriter writer, bool video_write);
  void DrawYoloDetections(cv::Mat &im_mat, VecBox &boxes,
                          bool console_show = true);
#endif
  void PrintYoloDetections(std::ofstream &file, VecBox &boxes, int count);
};

#endif // SHADOW_YOLO_H
