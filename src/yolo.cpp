#include "yolo.h"
#include "cl.h"
#include "image.h"
#include "kernel.h"
#include "parser.h"
#include "util.h"

#include <boost/algorithm/string.hpp>
#include <ctime>

using namespace std;

Yolo::Yolo(string cfgfile, string weightfile, float threshold) {
  cfgfile_ = cfgfile;
  weightfile_ = weightfile;
  threshold_ = threshold;
}
Yolo::~Yolo() {}

void Yolo::Setup() {
#ifdef USE_CL
  CL::CLSetup();
  Kernel::KernelSetup();
#endif
  Parser parser;
  parser.ParseNetworkCfg(net_, cfgfile_);
  if (weightfile_.empty())
    error("weight file is empty!");
  parser.LoadWeights(net_, weightfile_);
  class_num_ = net_.class_num_;
  grid_size_ = net_.grid_size_;
  sqrt_box_ = net_.sqrt_box_;
  box_num_ = net_.box_num_;
  out_num_ = net_.out_num_;
}
void Yolo::Release() {
  net_.ReleaseNetwork();
#ifdef USE_CL
  Kernel::KernelRelease();
  CL::CLRelease();
#endif
}

void Yolo::Test(string imagefile) {
  net_.SetBatchNetwork(1);

  clock_t time = clock();
  cv::Mat im = cv::imread(imagefile);
  vector<cv::Mat> images(1, im);
  vector<VecBox> Bboxes;
  PredictYoloDetections(images, Bboxes);
  Boxes::BoxesNMS(Bboxes[0], class_num_, 0.5);
  cout << "Predicted in " << (float)(clock() - time) / CLOCKS_PER_SEC
       << " seconds" << endl;

  DrawYoloDetections(im, Bboxes[0], true);
  cv::imshow("result", im);
  cv::waitKey();
}

void Yolo::BatchTest(string listfile, bool write) {
  net_.SetBatchNetwork(2);

  string outfile = listfile;
  boost::replace_last(outfile, ".", "-result.");

  vector<string> imagelist;
  Parser::LoadImageList(imagelist, listfile);
  size_t num_im = imagelist.size();

  clock_t time = clock();
  vector<cv::Mat> images;
  for (int i = 0; i < num_im; ++i) {
    images.push_back(cv::imread(imagelist[i]));
  }
  vector<VecBox> Bboxes;
  PredictYoloDetections(images, Bboxes);

  ofstream file(outfile);
  for (int j = 0; j < num_im; ++j) {
    Boxes::BoxesNMS(Bboxes[j], class_num_, 0.5);
    DrawYoloDetections(images[j], Bboxes[j], false);
    if (write) {
      string path = imagelist[j];
      boost::replace_last(path, ".", "-result.");
      cv::imwrite(path, images[j]);
    }
    PrintYoloDetections(file, Bboxes[j], j + 1);
    if (!((j + 1) % 100))
      cout << "Processing: " << j + 1 << " / " << num_im << endl;
  }
  cout << "Processing: " << num_im << " / " << num_im << endl;
  float sec = (float)(clock() - time) / CLOCKS_PER_SEC;
  cout << "Processed in: " << sec << " seconds, each frame: " << sec / num_im
       << "seconds" << endl;
}

void Yolo::VideoTest(string videofile, bool show) {
  net_.SetBatchNetwork(1);

  string outfile = videofile;
  boost::replace_last(outfile, ".", "-result.");

  cv::VideoCapture capture;
  if (!capture.open(videofile))
    exit(1);
  double format = capture.get(CV_CAP_PROP_FOURCC);
  double rate = capture.get(CV_CAP_PROP_FPS);
  double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
  double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  cv::VideoWriter writer(outfile, (int)format, rate,
                         cv::Size((int)width, (int)height));
  if (show) {
    cv::namedWindow("yolo video test!-:)", CV_WINDOW_NORMAL);
  }
  cv::Mat im;
  clock_t time;
  while (capture.read(im)) {
    if (im.empty())
      break;
    time = clock();
    vector<cv::Mat> images(1, im);
    vector<VecBox> Bboxes;
    PredictYoloDetections(images, Bboxes);
    Boxes::BoxesNMS(Bboxes[0], class_num_, 0.5);
    DrawYoloDetections(im, Bboxes[0], show);
    writer.write(im);
    float sec = (float)(clock() - time) / CLOCKS_PER_SEC;
    double waittime = 1000.0 * (1.0 / rate - sec);
    waittime = waittime > 1 ? waittime : 1;
    if (show) {
      cv::imshow("yolo video test!-:)", im);
      if ((cv::waitKey((int)waittime) % 256) == 27)
        break;
      cout << "FPS:" << 1.0 / sec << endl;
    }
  }
  capture.release();
  writer.release();
}

void Yolo::Demo(int camera, bool save) {
  net_.SetBatchNetwork(1);

  cv::VideoCapture capture;
  if (!capture.open(camera))
    exit(1);
  double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
  double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  cv::VideoWriter writer("./data/demo-result.avi",
                         CV_FOURCC('M', 'J', 'P', 'G'), 15,
                         cv::Size((int)width, (int)height));
  cv::namedWindow("yolo demo!-:)", CV_WINDOW_NORMAL);
  cv::Mat im;
  clock_t time;
  while (true) {
    capture.read(im);
    if (im.empty())
      break;
    time = clock();
    vector<cv::Mat> images(1, im);
    vector<VecBox> Bboxes;
    PredictYoloDetections(images, Bboxes);
    Boxes::BoxesNMS(Bboxes[0], class_num_, 0.5);
    DrawYoloDetections(im, Bboxes[0], true);
    if (save)
      writer.write(im);
    float sec = (float)(clock() - time) / CLOCKS_PER_SEC;
    double waittime = 1000.0 * (1 / 25.0 - sec);
    waittime = waittime > 1 ? waittime : 1;
    cv::imshow("yolo demo!-:)", im);
    if ((cv::waitKey((int)waittime) % 256) == 27)
      break;
    cout << "FPS:" << 1.0 / sec << endl;
  }
  capture.release();
  writer.release();
}

void Yolo::PredictYoloDetections(vector<cv::Mat> &images,
                                 vector<VecBox> &Bboxes) {
  size_t num_im = images.size();

  int batch = net_.batch_;
  size_t batch_off = num_im % batch;
  size_t batch_num = num_im / batch + (batch_off > 0 ? 1 : 0);

  cv::Mat sized;
  float *data = new float[batch * net_.in_num_];
  for (int count = 0, b = 1; b <= batch_num; ++b) {
    float *index = data;
    int c = 0;
    for (int i = count; i < b * batch && i < num_im; ++i, ++c) {
      cv::resize(images[i], sized, cv::Size(net_.in_w_, net_.in_h_));
      Image::GetFloatData(sized.data, net_.in_w_, net_.in_h_, net_.in_c_,
                          index);
      index += net_.in_num_;
    }
    float *predictions = net_.NetworkPredict(data);
    index = predictions;
    for (int i = 0; i < c; ++i) {
      VecBox boxes(grid_size_ * grid_size_ * box_num_);
      ConvertYoloDetections(index, class_num_, box_num_, sqrt_box_, grid_size_,
                            images[count + i].cols, images[count + i].rows,
                            boxes);
      Bboxes.push_back(boxes);
      index += out_num_;
    }
    count += c;
  }
}

void Yolo::ConvertYoloDetections(float *predictions, int classes, int box_num,
                                 int square, int side, int width, int height,
                                 VecBox &boxes) {
  for (int i = 0; i < side * side; ++i) {
    int row = i / side;
    int col = i % side;
    for (int n = 0; n < box_num; ++n) {
      int index = i * box_num + n;
      int p_index = side * side * classes + i * box_num + n;
      float scale = predictions[p_index];
      int box_index = side * side * (classes + box_num) + (i * box_num + n) * 4;

      float x, y, w, h;
      x = (predictions[box_index + 0] + col) / side * width;
      y = (predictions[box_index + 1] + row) / side * height;
      w = pow(predictions[box_index + 2], (square ? 2 : 1)) * width;
      h = pow(predictions[box_index + 3], (square ? 2 : 1)) * height;

      x = x - w / 2;
      y = y - h / 2;

      x = x > 0 ? x : 0;
      x = x < width ? x : (width - 1);
      y = y > 0 ? y : 0;
      y = y < height ? y : (height - 1);
      w = (x + w) < width ? w : (width - x - 1);
      h = (y + h) < height ? h : (height - y - 1);

      boxes[index].x = x;
      boxes[index].y = y;
      boxes[index].w = w;
      boxes[index].h = h;

      boxes[index].classindex = -1;
      boxes[index].probability.clear();
      boxes[index].probability.reserve(classes);
      for (int j = 0; j < classes; ++j) {
        int class_index = i * classes;
        float prob = scale * predictions[class_index + j];
        boxes[index].probability.push_back(prob > threshold_ ? prob : 0);
      }
    }
  }
}

void Yolo::DrawYoloDetections(cv::Mat &image, VecBox &boxes, bool show) {
  for (int b = 0; b < boxes.size(); ++b) {
    int classindex = boxes[b].classindex;
    vector<float> probability = boxes[b].probability;
    if (probability[classindex] < threshold_)
      continue;
    //    if (classindex != 0 && classindex != 5 && classindex != 6)
    //      continue;
    //    if (classindex != 0)
    //      continue;
    //    if (classindex != 5 && classindex != 6)
    //      continue;

    cv::Scalar scalar;
    if (classindex == 0)
      scalar = cv::Scalar(0, 255, 0);
    else
      scalar = cv::Scalar(255, 0, 0);

    Box box = boxes[b];
    cv::rectangle(image, cv::Point((int)box.x, (int)box.y),
                  cv::Point((int)(box.x + box.w), (int)(box.y + box.h)), scalar,
                  2, 8, 0);
    if (show) {
      cout << "x = " << box.x << ", y = " << box.y << ", w = " << box.w
           << ", h = " << box.h << ", score = " << probability[classindex]
           << endl;
    }
  }
}

void Yolo::PrintYoloDetections(std::ofstream &file, VecBox &boxes, int count) {
  for (int b = 0; b < boxes.size(); ++b) {
    int classindex = boxes[b].classindex;
    vector<float> probability = boxes[b].probability;
    if (probability[classindex] < threshold_)
      continue;

    Box box = boxes[b];
    file << count << ", " << box.x << ", " << box.y << ", " << box.w << ", "
         << box.h << ", " << probability[classindex] << endl;
  }
}
