#ifndef SHADOW_EXAMPLES_DEMO_DETECTION_HPP
#define SHADOW_EXAMPLES_DEMO_DETECTION_HPP

#include "detection_faster_rcnn.hpp"
#include "detection_mtcnn.hpp"
#include "detection_refinedet.hpp"
#include "detection_ssd.hpp"
#include "detection_yolo.hpp"

#include <memory>

namespace Shadow {

class DemoDetection {
 public:
  explicit DemoDetection(const std::string &method_name = "ssd") {
    if (method_name == "faster") {
      method_ = std::make_shared<DetectionFasterRCNN>();
    } else if (method_name == "mtcnn") {
      method_ = std::make_shared<DetectionMTCNN>();
    } else if (method_name == "ssd") {
      method_ = std::make_shared<DetectionSSD>();
    } else if (method_name == "refinedet") {
      method_ = std::make_shared<DetectionRefineDet>();
    } else if (method_name == "yolo") {
      method_ = std::make_shared<DetectionYOLO>();
    } else {
      LOG(FATAL) << "Unknown method " << method_name;
    }
  }

  void Setup(const std::string &model_file, const VecInt &in_shape) {
    method_->Setup(model_file, in_shape);
  }

  void Test(const std::string &image_file);
  void BatchTest(const std::string &list_file, bool image_write = false);
#if defined(USE_OpenCV)
  void VideoTest(const std::string &video_file, bool video_show = true,
                 bool video_write = false);
  void CameraTest(int camera, bool video_write = false);
#endif

  void Predict(const JImage &im_src, const VecRectF &rois,
               std::vector<VecBoxF> *Gboxes,
               std::vector<std::vector<VecPointF>> *Gpoints) {
    method_->Predict(im_src, rois, Gboxes, Gpoints);
  }
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<VecBoxF> *Gboxes,
               std::vector<std::vector<VecPointF>> *Gpoints) {
    method_->Predict(im_mat, rois, Gboxes, Gpoints);
  }
#endif

 private:
#if defined(USE_OpenCV)
  void CaptureTest(cv::VideoCapture *capture, const std::string &window_name,
                   bool video_show, cv::VideoWriter *writer);
  void DrawDetections(const VecBoxF &boxes, cv::Mat *im_mat,
                      bool console_show = true);
#endif

  void PrintDetections(const std::string &im_name, const VecBoxF &boxes,
                       std::ostream *os);

  std::shared_ptr<Method> method_ = nullptr;
  JImage im_ini_;
  std::vector<VecBoxF> Gboxes_;
  std::vector<std::vector<VecPointF>> Gpoints_;
  Timer timer_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DEMO_DETECTION_HPP
