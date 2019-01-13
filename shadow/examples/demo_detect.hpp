#ifndef SHADOW_EXAMPLES_DEMO_DETECT_HPP
#define SHADOW_EXAMPLES_DEMO_DETECT_HPP

#include "method.hpp"

namespace Shadow {

class DemoDetect {
 public:
  explicit DemoDetect(const std::string &method_name);

  void Setup(const std::string &model_file) { method_->Setup(model_file); }

  void Test(const std::string &image_file);
  void BatchTest(const std::string &list_file, bool image_write = false);
#if defined(USE_OpenCV)
  void VideoTest(const std::string &video_file, bool video_show = true,
                 bool video_write = false);
  void CameraTest(int camera, bool video_write = false);
#endif

 private:
#if defined(USE_OpenCV)
  void CaptureTest(cv::VideoCapture *capture, const std::string &window_name,
                   bool video_show, cv::VideoWriter *writer);
  void DrawDetections(const VecBoxF &boxes, cv::Mat *im_mat);
#endif

  void DrawDetections(const VecBoxF &boxes, JImage *im_src);

  void PrintConsole(const VecBoxF &boxes, bool split = false);

  void PrintStream(const std::string &im_name, const VecBoxF &boxes,
                   std::ostream *os);

  std::shared_ptr<Method> method_ = nullptr;
  JImage im_ini_;
  VecBoxF boxes_;
  std::vector<VecPointF> Gpoints_;
  Timer timer_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DEMO_DETECT_HPP
