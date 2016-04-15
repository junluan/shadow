#include "yolo.h"
#include "kernel.h"
#include "parser.h"
#include "util.h"

#include <ctime>

using namespace std;

Yolo::Yolo(string cfgfile, string weightfile, float threshold) {
  cfgfile_ = cfgfile;
  weightfile_ = weightfile;
  threshold_ = threshold;
}

Yolo::~Yolo() {}

void Yolo::Setup() {
  Kernel::KernelSetup();

  Parser parser;
  parser.ParseNetworkCfg(net_, cfgfile_);
  if (weightfile_.empty())
    error("Weight file is empty!");
  parser.LoadWeights(net_, weightfile_);
  class_num_ = net_.class_num_;
  grid_size_ = net_.grid_size_;
  sqrt_box_ = net_.sqrt_box_;
  box_num_ = net_.box_num_;
  out_num_ = net_.out_num_;
  batch_data_ = new float[net_.batch_ * net_.in_num_];
  im_res_ = new JImage(3, net_.in_h_, net_.in_w_);
}

void Yolo::Release() {
  net_.ReleaseNetwork();
  Kernel::KernelRelease();
}

void Yolo::Test(string imagefile) {
  net_.SetNetworkBatch(1);

  clock_t time = clock();
  JImage *img = new JImage();
  img->Read(imagefile);
  vector<JImage *> images(1, img);
  vector<VecBox> Bboxes;
  PredictYoloDetections(images, Bboxes);
  Boxes::BoxesNMS(Bboxes[0], 0.5);
  cout << "Predicted in " << static_cast<float>(clock() - time) / CLOCKS_PER_SEC
       << " seconds" << endl;
  img->Rectangle(Bboxes[0]);
  img->Show("result");
}

void Yolo::BatchTest(string listfile, bool image_write) {
  net_.SetNetworkBatch(2);

  string outfile = find_replace_last(listfile, ".", "-result.");

  vector<string> imagelist;
  Parser::LoadImageList(imagelist, listfile);
  size_t num_im = imagelist.size();

  clock_t time = clock();
  vector<JImage *> images;
  for (int i = 0; i < num_im; ++i) {
    JImage *img = new JImage();
    img->Read(imagelist[i]);
    images.push_back(img);
  }
  vector<VecBox> Bboxes;
  PredictYoloDetections(images, Bboxes);

  ofstream file(outfile);
  for (int j = 0; j < num_im; ++j) {
    Boxes::BoxesNMS(Bboxes[j], 0.5);
    if (image_write) {
      string path = find_replace_last(imagelist[j], ".", "-result.");
      images[j]->Rectangle(Bboxes[j], false);
      images[j]->Write(path);
    }
    PrintYoloDetections(file, Bboxes[j], j + 1);
    if (!((j + 1) % 100))
      cout << "Processing: " << j + 1 << " / " << num_im << endl;
  }
  cout << "Processing: " << num_im << " / " << num_im << endl;
  float sec = static_cast<float>(clock() - time) / CLOCKS_PER_SEC;
  cout << "Processed in: " << sec << " seconds, each frame: " << sec / num_im
       << "seconds" << endl;
}

#ifdef USE_OpenCV
void Yolo::VideoTest(string videofile, bool video_show) {
  net_.SetNetworkBatch(1);

  string outfile = find_replace_last(videofile, ".", "-result.");

  cv::VideoCapture capture;
  if (!capture.open(videofile))
    error("error when opening video file " + videofile);
  int format = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
  float rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer(outfile, format, rate, cv::Size(width, height));
  if (video_show) {
    cv::namedWindow("yolo video test!-:)", CV_WINDOW_NORMAL);
  }
  cv::Mat im_mat;
  JImage *img = new JImage(3, height, width);
  vector<JImage *> images(1, img);
  clock_t time;
  VecBox currBoxes;
  while (capture.read(im_mat)) {
    if (im_mat.empty())
      break;
    time = clock();
    images[0]->FromMat(im_mat);
    vector<VecBox> Bboxes;
    PredictYoloDetections(images, Bboxes);
    Boxes::BoxesNMS(Bboxes[0], 0.5);
    Boxes::SmoothBoxes(currBoxes, Bboxes[0], 0.3);
    currBoxes = Bboxes[0];
    DrawYoloDetections(im_mat, Bboxes[0], true);
    writer.write(im_mat);
    float sec = static_cast<float>(clock() - time) / CLOCKS_PER_SEC;
    int waittime = static_cast<int>(1000.0f * (1.0f / rate - sec));
    waittime = waittime > 1 ? waittime : 1;
    if (video_show) {
      cv::imshow("yolo video test!-:)", im_mat);
      if ((cv::waitKey(waittime) % 256) == 27)
        break;
      cout << "FPS:" << 1.0 / sec << endl;
    }
  }
  capture.release();
  writer.release();
}

void Yolo::Demo(int camera, bool video_write) {
  net_.SetNetworkBatch(1);

  cv::VideoCapture capture;
  if (!capture.open(camera))
    error("error when opening camera!");
  int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer("./data/demo-result.avi",
                         CV_FOURCC('M', 'J', 'P', 'G'), 15,
                         cv::Size(width, height));
  cv::namedWindow("yolo demo!-:)", CV_WINDOW_NORMAL);
  cv::Mat im_mat;
  JImage *img = new JImage(3, height, width);
  vector<JImage *> images(1, img);
  clock_t time;
  VecBox currBoxes;
  while (true) {
    capture.read(im_mat);
    if (im_mat.empty())
      break;
    time = clock();
    images[0]->FromMat(im_mat);
    vector<VecBox> Bboxes;
    PredictYoloDetections(images, Bboxes);
    Boxes::BoxesNMS(Bboxes[0], 0.5);
    Boxes::SmoothBoxes(currBoxes, Bboxes[0], 0.8);
    currBoxes = Bboxes[0];
    DrawYoloDetections(im_mat, Bboxes[0], true);
    if (video_write)
      writer.write(im_mat);
    float sec = static_cast<float>(clock() - time) / CLOCKS_PER_SEC;
    int waittime = static_cast<int>(1000.0f * (1.0f / 25.0f - sec));
    waittime = waittime > 1 ? waittime : 1;
    cv::imshow("yolo demo!-:)", im_mat);
    if ((cv::waitKey(waittime) % 256) == 27)
      break;
    cout << "FPS:" << 1.0 / sec << endl;
  }
  capture.release();
  writer.release();
}
#endif

void Yolo::PredictYoloDetections(std::vector<JImage *> &images,
                                 std::vector<VecBox> &Bboxes) {
  size_t num_im = images.size();

  int batch = net_.batch_;
  size_t batch_off = num_im % batch;
  size_t batch_num = num_im / batch + (batch_off > 0 ? 1 : 0);

  float *index = NULL;
  for (int count = 0, b = 1; b <= batch_num; ++b) {
    index = batch_data_;
    int c = 0;
    for (int i = count; i < b * batch && i < num_im; ++i, ++c) {
      images[i]->Resize(im_res_, net_.in_h_, net_.in_w_);
      im_res_->SetBatchData(index);
      index += net_.in_num_;
    }
    predictions_ = net_.PredictNetwork(batch_data_);
    index = predictions_;
    for (int i = 0; i < c; ++i) {
      VecBox boxes(grid_size_ * grid_size_ * box_num_);
      ConvertYoloDetections(index, class_num_, box_num_, sqrt_box_, grid_size_,
                            images[count + i]->w_, images[count + i]->h_,
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
      x = (predictions[box_index + 0] + col) / side;
      y = (predictions[box_index + 1] + row) / side;
      w = pow(predictions[box_index + 2], (square ? 2 : 1));
      h = pow(predictions[box_index + 3], (square ? 2 : 1));

      x = constrain(0.f, 1.f, x - w / 2);
      y = constrain(0.f, 1.f, y - h / 2);
      w = constrain(0.f, 1.f, x + w) - x;
      h = constrain(0.f, 1.f, y + h) - y;

      boxes[index].x = x * width;
      boxes[index].y = y * height;
      boxes[index].w = w * width;
      boxes[index].h = h * height;

      float max_score = 0;
      int max_index = -1;
      for (int j = 0; j < classes; ++j) {
        float score = scale * predictions[i * classes + j];
        if (score > threshold_ && score > max_score) {
          max_score = score;
          max_index = j;
        }
      }
      boxes[index].score = max_score;
      boxes[index].class_index = max_index;
    }
  }
}

#ifdef USE_OpenCV
void Yolo::DrawYoloDetections(cv::Mat &im_mat, VecBox &boxes,
                              bool console_show) {
  for (int b = 0; b < boxes.size(); ++b) {
    int classindex = boxes[b].class_index;
    if (classindex == -1)
      continue;

    cv::Scalar scalar;
    if (classindex == 0)
      scalar = cv::Scalar(0, 255, 0);
    else
      scalar = cv::Scalar(255, 0, 0);

    Box box = boxes[b];
    cv::rectangle(im_mat,
                  cv::Point(static_cast<int>(box.x), static_cast<int>(box.y)),
                  cv::Point(static_cast<int>(box.x + box.w),
                            static_cast<int>(box.y + box.h)),
                  scalar, 2, 8, 0);
    if (console_show) {
      std::cout << "x = " << box.x << ", y = " << box.y << ", w = " << box.w
                << ", h = " << box.h << ", score = " << box.score << std::endl;
    }
  }
}
#endif

void Yolo::PrintYoloDetections(std::ofstream &file, VecBox &boxes, int count) {
  for (int b = 0; b < boxes.size(); ++b) {
    Box box = boxes[b];
    if (box.class_index == -1)
      continue;
    file << count << ", " << box.x << ", " << box.y << ", " << box.w << ", "
         << box.h << ", " << box.score << endl;
  }
}
