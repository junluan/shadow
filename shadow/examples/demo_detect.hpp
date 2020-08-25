#ifndef SHADOW_EXAMPLES_DEMO_DETECT_HPP_
#define SHADOW_EXAMPLES_DEMO_DETECT_HPP_

#include "algorithm/method.hpp"

#include <memory>

namespace Shadow {

class DemoDetect {
 public:
  explicit DemoDetect(const std::string& method_name);
  ~DemoDetect() { LOG(INFO) << profiler_.get_stats_str(); }

  void Setup(const std::string& model_file) { method_->Setup(model_file); }

  void Test(const std::string& image_file);
  void BatchTest(const std::string& list_file, bool image_write = false);
  void VideoTest(const std::string& video_file, bool video_show = true,
                 bool video_write = false);
  void CameraTest(int camera, bool video_write = false);

 private:
  void CaptureTest(cv::VideoCapture* capture, const std::string& window_name,
                   bool video_show, cv::VideoWriter* writer);

  static void DrawDetections(const VecBoxF& boxes, cv::Mat* im_mat);

  static void PrintConsole(const VecBoxF& boxes, bool split = false);

  static void PrintStream(const std::string& im_name, const VecBoxF& boxes,
                          std::ostream* os);

  Profiler profiler_;
  std::shared_ptr<Method> method_ = nullptr;
  VecBoxF boxes_;
  std::vector<VecPointF> Gpoints_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DEMO_DETECT_HPP_
