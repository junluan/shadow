#include "demo.hpp"

#include "shadow/util/jimage_proc.hpp"

void Demo::Test(const std::string &image_file) {
  im_ini_.Read(image_file);
  VecRectF rois;
  rois.push_back(RectF(0, 0, im_ini_.w_, im_ini_.h_));
  timer_.start();
  Predict(im_ini_, rois, &Bboxes_);
  Boxes::Amend(&Bboxes_, rois);
  auto boxes = Boxes::NMS(Bboxes_, 0.5);
  std::cout << "Predicted in " << timer_.get_millisecond() << " ms"
            << std::endl;

  for (auto &boxF : boxes) {
    BoxI box(boxF);
    Scalar scalar(0, 255, 255);
    if (box.label == 1) {
      scalar = Scalar(0, 255, 0);
    } else if (box.label == 2) {
      scalar = Scalar(0, 0, 255);
    }
    JImageProc::Rectangle(&im_ini_, box.RectInt(), scalar);
    std::cout << box.score << ", " << box.xmin << ", " << box.ymin << ", "
              << box.xmax << ", " << box.ymax << std::endl;
  }
  im_ini_.Show("result");
}

void Demo::BatchTest(const std::string &list_file, bool image_write) {
  const VecString &image_list = Util::load_list(list_file);
  size_t num_im = image_list.size();

  double time_cost = 0;
  std::ofstream file(Util::find_replace_last(list_file, ".", "-result."));
  int count = 0;
  for (auto &im_path : image_list) {
    im_ini_.Read(im_path);
    VecRectF rois;
    rois.push_back(RectF(0, 0, im_ini_.w_, im_ini_.h_));
    timer_.start();
    Predict(im_ini_, rois, &Bboxes_);
    Boxes::Amend(&Bboxes_, rois, im_ini_.h_, im_ini_.w_);
    auto boxes = Boxes::NMS(Bboxes_, 0.5);
    time_cost += timer_.get_millisecond();
    if (image_write) {
      const std::string &path =
          Util::find_replace_last(im_path, ".", "-result.");
      for (auto &box : boxes) {
        Scalar scalar(0, 255, 255);
        if (box.label == 1) {
          scalar = Scalar(0, 255, 0);
        } else if (box.label == 2) {
          scalar = Scalar(0, 0, 255);
        }
        JImageProc::Rectangle(&im_ini_, box.RectInt(), scalar);
      }
      im_ini_.Write(path);
    }
    PrintDetections(im_path, boxes, &file);
    count++;
    if (!((count) % 100)) {
      std::cout << "Processing: " << count << " / " << num_im << std::endl;
    }
  }
  file.close();
  std::cout << "Processing: " << num_im << " / " << num_im << std::endl;
  std::cout << "Processed in: " << time_cost
            << " ms, each frame: " << time_cost / num_im << " ms" << std::endl;
}

#if defined(USE_OpenCV)
void Demo::VideoTest(const std::string &video_file, bool video_show,
                     bool video_write) {
  cv::VideoCapture capture;
  if (!capture.open(video_file)) {
    Fatal("error when opening video file " + video_file);
  }
  float rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer;
  if (video_write) {
    const auto &outfile = Util::change_extension(video_file, "-result.avi");
    int format = CV_FOURCC('H', '2', '6', '4');
    writer.open(outfile, format, rate, cv::Size(width, height));
  }
  CaptureTest(&capture, "video test!-:)", video_show, &writer);
  capture.release();
  writer.release();
}

void Demo::CameraTest(int camera, bool video_write) {
  cv::VideoCapture capture;
  if (!capture.open(camera)) {
    Fatal("error when opening camera!");
  }
  float rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer;
  if (video_write) {
    const auto &outfile = "data/demo-result.avi";
    int format = CV_FOURCC('H', '2', '6', '4');
    writer.open(outfile, format, rate, cv::Size(width, height));
  }
  CaptureTest(&capture, "camera test!-:)", true, &writer);
  capture.release();
  writer.release();
}
#endif

#if defined(USE_OpenCV)
void Demo::CaptureTest(cv::VideoCapture *capture,
                       const std::string &window_name, bool video_show,
                       cv::VideoWriter *writer) {
  float rate = static_cast<float>(capture->get(CV_CAP_PROP_FPS));
  if (video_show) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  }
  cv::Mat im_mat;
  VecBoxF boxes, oldBoxes;
  while (capture->read(im_mat)) {
    if (im_mat.empty()) break;
    im_ini_.FromMat(im_mat);
    VecRectF rois;
    rois.push_back(RectF(0, 0, im_ini_.w_, im_ini_.h_));
    timer_.start();
    Predict(im_ini_, rois, &Bboxes_);
    Boxes::Amend(&Bboxes_, rois, im_ini_.h_, im_ini_.w_);
    boxes = Boxes::NMS(Bboxes_, 0.5);
    Boxes::Smooth(oldBoxes, &boxes, 0.3);
    double time_cost = timer_.get_millisecond();
    oldBoxes = boxes;
    DrawDetections(boxes, &im_mat, true);
    if (writer->isOpened()) {
      writer->write(im_mat);
    }
    int wait_time = static_cast<int>(1000.0f / rate - time_cost) - 1;
    wait_time = wait_time > 1 ? wait_time : 1;
    if (video_show) {
      cv::imshow(window_name, im_mat);
      if (cv::waitKey(wait_time) % 256 == 32) {
        if ((cv::waitKey(0) % 256) == 27) break;
      }
      std::cout << "FPS:" << 1000.0f / time_cost << std::endl;
    }
  }
}

void Demo::DrawDetections(const VecBoxF &boxes, cv::Mat *im_mat,
                          bool console_show) {
  for (auto &boxF : boxes) {
    const BoxI box(boxF);
    cv::Scalar scalar(0, 255, 255);
    if (box.label == 1) {
      scalar = cv::Scalar(0, 255, 0);
    } else if (box.label == 2) {
      scalar = cv::Scalar(255, 0, 0);
    } else {
      continue;
    }
    cv::rectangle(*im_mat, cv::Point(box.xmin, box.ymin),
                  cv::Point(box.xmax, box.ymax), scalar, 2);
    if (console_show) {
      std::cout << "xmin = " << box.xmin << ", ymin = " << box.ymin
                << ", xmax = " << box.xmax << ", ymax = " << box.ymax
                << ", label = " << box.label << ", score = " << box.score
                << std::endl;
    }
  }
}
#endif

void Demo::PrintDetections(const std::string &im_name, const VecBoxF &boxes,
                           std::ofstream *file) {
  for (auto &boxF : boxes) {
    const BoxI box(boxF);
    *file << im_name << " " << box.xmin << ", " << box.ymin << ", " << box.xmax
          << ", " << box.ymax << ", " << box.label << ", " << box.score
          << std::endl;
  }
}
