#ifndef SHADOW_YOLO_H
#define SHADOW_YOLO_H

#include "boxes.h"
#include "image.h"
#include "network.h"

#include <fstream>
#include <string>

class Yolo {
public:
  Yolo(std::string cfgfile, std::string weightfile, float threshold);
  ~Yolo();

  void Setup();
  void Test(std::string imagefile);
  void BatchTest(std::string listfile, bool image_write = false);
  void VideoTest(std::string videofile, bool video_show = false);
  void Demo(int camera, bool video_write = false);
  void Release();

private:
  std::string cfgfile_, weightfile_;
  float threshold_;
  Network net_;
  int class_num_, grid_size_, box_num_, sqrt_box_, out_num_;

  void PredictYoloDetections(std::vector<image> &images,
                             std::vector<VecBox> &Bboxes);
  void ConvertYoloDetections(float *predictions, int classes, int num,
                             int square, int side, int width, int height,
                             VecBox &boxes);
  void PrintYoloDetections(std::ofstream &file, VecBox &boxes, int count);
};

#endif // SHADOW_YOLO_H
