#include "demo_detection.hpp"
#include "util/jimage_proc.hpp"

namespace Shadow {

void DemoDetection::Test(const std::string &image_file) {
  im_ini_.Read(image_file);
  timer_.start();
  Predict(im_ini_, RectF(0, 0, im_ini_.w_, im_ini_.h_), &boxes_, &Gpoints_);
  boxes_ = Boxes::NMS(boxes_, 0.5);
  LOG(INFO) << "Predicted in " << timer_.get_millisecond() << " ms";
  for (const auto &boxF : boxes_) {
    const BoxI box(boxF);
    int color_r = (box.label * 100) % 255;
    int color_g = (color_r + 100) % 255;
    int color_b = (color_g + 100) % 255;
    Scalar scalar(color_r, color_g, color_b);
    JImageProc::Rectangle(&im_ini_, box.RectInt(), scalar);
    LOG(INFO) << "xmin = " << box.xmin << ", ymin = " << box.ymin
              << ", xmax = " << box.xmax << ", ymax = " << box.ymax
              << ", label = " << box.label << ", score = " << box.score;
  }
  im_ini_.Show("result");
}

void DemoDetection::BatchTest(const std::string &list_file, bool image_write) {
  const auto &image_list = Util::load_list(list_file);
  int num_im = static_cast<int>(image_list.size()), count = 0;
  double time_cost = 0;
  Process process(20, num_im, "Processing: ");
  const auto &result_file = Util::find_replace_last(list_file, ".", "-result.");
  std::ofstream file(result_file);
  CHECK(file.is_open()) << "Can't open file " << result_file;
  for (const auto &im_path : image_list) {
    im_ini_.Read(im_path);
    timer_.start();
    Predict(im_ini_, RectF(0, 0, im_ini_.w_, im_ini_.h_), &boxes_, &Gpoints_);
    boxes_ = Boxes::NMS(boxes_, 0.5);
    time_cost += timer_.get_millisecond();
    if (image_write) {
      const auto &out_file = Util::find_replace_last(im_path, ".", "-result.");
      for (const auto &boxF : boxes_) {
        const BoxI box(boxF);
        int color_r = (box.label * 100) % 255;
        int color_g = (color_r + 100) % 255;
        int color_b = (color_g + 100) % 255;
        Scalar scalar(color_r, color_g, color_b);
        JImageProc::Rectangle(&im_ini_, box.RectInt(), scalar);
      }
      im_ini_.Write(out_file);
    }
    PrintDetections(im_path, boxes_, &file);
    process.update(count++, &std::cout);
  }
  file.close();
  LOG(INFO) << "Processed in: " << time_cost
            << " ms, each frame: " << time_cost / num_im << " ms";
}

#if defined(USE_OpenCV)
void DemoDetection::VideoTest(const std::string &video_file, bool video_show,
                              bool video_write) {
  cv::VideoCapture capture;
  CHECK(capture.open(video_file))
      << "Error when opening video file " << video_file;
  auto rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  auto width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  auto height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer;
  if (video_write) {
    const auto &out_file = Util::change_extension(video_file, "-result.avi");
    int format = CV_FOURCC('H', '2', '6', '4');
    writer.open(out_file, format, rate, cv::Size(width, height));
  }
  CaptureTest(&capture, "video test!-:)", video_show, &writer);
  capture.release();
  writer.release();
}

void DemoDetection::CameraTest(int camera, bool video_write) {
  cv::VideoCapture capture;
  CHECK(capture.open(camera)) << "Error when opening camera!";
  auto rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  auto width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  auto height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer;
  if (video_write) {
    const auto &out_file = "data/demo-result.avi";
    int format = CV_FOURCC('H', '2', '6', '4');
    writer.open(out_file, format, rate, cv::Size(width, height));
  }
  CaptureTest(&capture, "camera test!-:)", true, &writer);
  capture.release();
  writer.release();
}

void DemoDetection::CaptureTest(cv::VideoCapture *capture,
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
    im_ini_.FromMat(im_mat);
    timer_.start();
    Predict(im_ini_, RectF(0, 0, im_ini_.w_, im_ini_.h_), &boxes_, &Gpoints_);
    boxes_ = Boxes::NMS(boxes_, 0.5);
    time_cost = timer_.get_millisecond();
    Boxes::Smooth(old_boxes_, &boxes_, 0.3);
    old_boxes_ = boxes_;
    DrawDetections(boxes_, &im_mat, true);
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

void DemoDetection::DrawDetections(const VecBoxF &boxes, cv::Mat *im_mat,
                                   bool console_show) {
  for (const auto &boxF : boxes) {
    const BoxI box(boxF);
    int color_r = (box.label * 100) % 255;
    int color_g = (color_r + 100) % 255;
    int color_b = (color_g + 100) % 255;
    cv::Scalar scalar(color_b, color_g, color_r);
    cv::rectangle(*im_mat, cv::Point(box.xmin, box.ymin),
                  cv::Point(box.xmax, box.ymax), scalar, 2);
    if (console_show) {
      LOG(INFO) << "xmin = " << box.xmin << ", ymin = " << box.ymin
                << ", xmax = " << box.xmax << ", ymax = " << box.ymax
                << ", label = " << box.label << ", score = " << box.score;
    }
  }
}
#endif

void DemoDetection::PrintDetections(const std::string &im_name,
                                    const VecBoxF &boxes, std::ostream *os) {
  *os << im_name << ":" << std::endl;
  *os << "objects:" << std::endl;
  for (const auto &boxF : boxes) {
    const BoxI box(boxF);
    *os << "   " << box.xmin << " " << box.ymin << " " << box.xmax << " "
        << box.ymax << " " << box.label << " " << box.score << std::endl;
  }
  *os << "-------------------------" << std::endl;
}

}  // namespace Shadow
