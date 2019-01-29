#include "demo_detect.hpp"

#include "detect_faster_rcnn.hpp"
#include "detect_mtcnn.hpp"
#include "detect_refinedet.hpp"
#include "detect_ssd.hpp"
#include "detect_yolo.hpp"

#include "util/jimage_proc.hpp"

namespace Shadow {

DemoDetect::DemoDetect(const std::string &method_name) {
  if (method_name == "faster") {
    method_ = std::make_shared<DetectFasterRCNN>();
  } else if (method_name == "mtcnn") {
    method_ = std::make_shared<DetectMTCNN>();
  } else if (method_name == "ssd") {
    method_ = std::make_shared<DetectSSD>();
  } else if (method_name == "refinedet") {
    method_ = std::make_shared<DetectRefineDet>();
  } else if (method_name == "yolo") {
    method_ = std::make_shared<DetectYOLO>();
  } else {
    LOG(FATAL) << "Unknown method " << method_name;
  }
}

void DemoDetect::Test(const std::string &image_file) {
  im_ini_.Read(image_file);
  timer_.start();
  method_->Predict(im_ini_, RectF(0, 0, im_ini_.w_, im_ini_.h_), &boxes_,
                   &Gpoints_);
  boxes_ = Boxes::NMS(boxes_, 0.5);
  LOG(INFO) << "Predicted in " << timer_.get_millisecond() << " ms";
  PrintConsole(boxes_);
  DrawDetections(boxes_, &im_ini_);
  im_ini_.Show("result");
}

void DemoDetect::BatchTest(const std::string &list_file, bool image_write) {
  const auto &image_list = Util::load_list(list_file);
  int num_im = static_cast<int>(image_list.size()), count = 0;
  double time_cost = 0;
  ProcessBar process_bar(20, num_im, "Processing: ");
  const auto &result_file = Util::find_replace_last(list_file, ".", "-result.");
  std::ofstream file(result_file);
  CHECK(file.is_open()) << "Can't open file " << result_file;
  for (const auto &im_path : image_list) {
    im_ini_.Read(im_path);
    timer_.start();
    method_->Predict(im_ini_, RectF(0, 0, im_ini_.w_, im_ini_.h_), &boxes_,
                     &Gpoints_);
    boxes_ = Boxes::NMS(boxes_, 0.5);
    time_cost += timer_.get_millisecond();
    PrintStream(im_path, boxes_, &file);
    if (image_write) {
      const auto &out_file = Util::find_replace_last(im_path, ".", "-result.");
      DrawDetections(boxes_, &im_ini_);
      im_ini_.Write(out_file);
    }
    process_bar.update(count++, &std::cout);
  }
  file.close();
  LOG(INFO) << "Processed in: " << time_cost
            << " ms, each frame: " << time_cost / num_im << " ms";
}

#if defined(USE_OpenCV)
#if CV_MAJOR_VERSION >= 4
#define CV_FOURCC cv::VideoWriter::fourcc
#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#endif

void DemoDetect::VideoTest(const std::string &video_file, bool video_show,
                           bool video_write) {
  cv::VideoCapture capture;
  CHECK(capture.open(video_file))
      << "Error when opening video file " << video_file;
  auto rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  auto width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  auto height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer;
  if (video_write) {
    const auto &out_file = Util::change_extension(video_file, "-result.mp4");
    int format = CV_FOURCC('H', '2', '6', '4');
    writer.open(out_file, format, rate, cv::Size(width, height));
  }
  CaptureTest(&capture, "video test!-:)", video_show, &writer);
  capture.release();
  writer.release();
}

void DemoDetect::CameraTest(int camera, bool video_write) {
  cv::VideoCapture capture;
  CHECK(capture.open(camera)) << "Error when opening camera!";
  auto rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  auto width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  auto height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer;
  if (video_write) {
    const auto &out_file = "data/demo-result.mp4";
    int format = CV_FOURCC('H', '2', '6', '4');
    writer.open(out_file, format, rate, cv::Size(width, height));
  }
  CaptureTest(&capture, "camera test!-:)", true, &writer);
  capture.release();
  writer.release();
}

void DemoDetect::CaptureTest(cv::VideoCapture *capture,
                             const std::string &window_name, bool video_show,
                             cv::VideoWriter *writer) {
  if (video_show) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  }
  cv::Mat im_mat;
  double time_sum = 0, time_cost;
  int count = 0;
  std::stringstream ss;
  ss.precision(5);
  while (capture->read(im_mat) && !im_mat.empty()) {
    timer_.start();
    method_->Predict(im_mat, RectF(0, 0, im_mat.cols, im_mat.rows), &boxes_,
                     &Gpoints_);
    boxes_ = Boxes::NMS(boxes_, 0.5);
    time_cost = timer_.get_millisecond();
    PrintConsole(boxes_, true);
    if (writer->isOpened() || video_show) {
      DrawDetections(boxes_, &im_mat);
    }
    if (writer->isOpened()) {
      writer->write(im_mat);
    }
    if (video_show) {
      time_sum += time_cost;
      if ((++count % 30) == 0) {
        ss.str("");
        ss << "FPS: " << 1000 * count / time_sum;
        count = 0, time_sum = 0;
      }
      cv::putText(im_mat, ss.str(), cv::Point(10, 30), cv::FONT_ITALIC, 0.7,
                  cv::Scalar(225, 105, 65), 2);
      cv::imshow(window_name, im_mat);
      if (cv::waitKey(1) % 256 == 32) {
        if ((cv::waitKey(0) % 256) == 27) break;
      }
    }
  }
}

void DemoDetect::DrawDetections(const VecBoxF &boxes, cv::Mat *im_mat) {
  for (const auto &boxF : boxes) {
    const BoxI box(boxF);
    int color_r = (box.label * 100) % 255;
    int color_g = (color_r + 100) % 255;
    int color_b = (color_g + 100) % 255;
    cv::Scalar scalar(color_b, color_g, color_r);
    cv::rectangle(*im_mat, cv::Point(box.xmin, box.ymin),
                  cv::Point(box.xmax, box.ymax), scalar, 2);
  }
}
#endif

void DemoDetect::DrawDetections(const Shadow::VecBoxF &boxes,
                                Shadow::JImage *im_src) {
  for (const auto &box : boxes_) {
    int color_r = (box.label * 100) % 255;
    int color_g = (color_r + 100) % 255;
    int color_b = (color_g + 100) % 255;
    Scalar scalar(color_r, color_g, color_b);
    JImageProc::Rectangle(im_src, box.RectInt(), scalar);
  }
}

void DemoDetect::PrintConsole(const Shadow::VecBoxF &boxes, bool split) {
  for (const auto &boxF : boxes) {
    const BoxI box(boxF);
    LOG(INFO) << "xmin = " << box.xmin << ", ymin = " << box.ymin
              << ", xmax = " << box.xmax << ", ymax = " << box.ymax
              << ", label = " << box.label << ", score = " << box.score;
  }
  if (split) {
    LOG(INFO) << "------------------------------";
  }
}

void DemoDetect::PrintStream(const std::string &im_name, const VecBoxF &boxes,
                             std::ostream *os) {
  *os << im_name << ":" << std::endl;
  *os << "objects:" << std::endl;
  for (const auto &boxF : boxes) {
    const BoxI box(boxF);
    *os << "   " << box.xmin << " " << box.ymin << " " << box.xmax << " "
        << box.ymax << " " << box.label << " " << box.score << std::endl;
  }
  *os << "------------------------------" << std::endl;
}

}  // namespace Shadow
