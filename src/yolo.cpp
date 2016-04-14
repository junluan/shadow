#include "yolo.h"
#include "image.h"
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
}

void Yolo::Release() {
  net_.ReleaseNetwork();
  Kernel::KernelRelease();
}

void Yolo::Test(string imagefile) {
  net_.SetNetworkBatch(1);

  clock_t time = clock();
  image img = Image::ImRead(imagefile);
  vector<image> images(1, img);
  vector<VecBox> Bboxes;
  PredictYoloDetections(images, Bboxes);
  Boxes::BoxesNMS(Bboxes[0], 0.5);
  cout << "Predicted in " << static_cast<float>(clock() - time) / CLOCKS_PER_SEC
       << " seconds" << endl;

  Image::ImRectangle(img, Bboxes[0]);
  Image::ImShow("result", img);
}

void Yolo::BatchTest(string listfile, bool image_write) {
  net_.SetNetworkBatch(2);

  string outfile = find_replace_last(listfile, ".", "-result.");

  vector<string> imagelist;
  Parser::LoadImageList(imagelist, listfile);
  size_t num_im = imagelist.size();

  clock_t time = clock();
  vector<image> images;
  for (int i = 0; i < num_im; ++i) {
    images.push_back(Image::ImRead(imagelist[i]));
  }
  vector<VecBox> Bboxes;
  PredictYoloDetections(images, Bboxes);

  ofstream file(outfile);
  for (int j = 0; j < num_im; ++j) {
    Boxes::BoxesNMS(Bboxes[j], 0.5);
    if (image_write) {
      string path = find_replace_last(imagelist[j], ".", "-result.");
      Image::ImRectangle(images[j], Bboxes[j], false);
      Image::ImWrite(path, images[j]);
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
  image im = Image::MakeImage(3, height, width);
  clock_t time;
  VecBox currBoxes;
  while (capture.read(im_mat)) {
    if (im_mat.empty())
      break;
    time = clock();
    Image::MatToImage(im_mat, im);
    vector<image> images(1, im);
    vector<VecBox> Bboxes;
    PredictYoloDetections(images, Bboxes);
    Boxes::BoxesNMS(Bboxes[0], 0.5);
    Boxes::SmoothBoxes(currBoxes, Bboxes[0], 0.3);
    currBoxes = Bboxes[0];
    Image::ImRectangle(im_mat, Bboxes[0], true);
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
  image im = Image::MakeImage(3, height, width);
  clock_t time;
  VecBox currBoxes;
  while (true) {
    capture.read(im_mat);
    if (im_mat.empty())
      break;
    time = clock();
    Image::MatToImage(im_mat, im);
    vector<image> images(1, im);
    vector<VecBox> Bboxes;
    PredictYoloDetections(images, Bboxes);
    Boxes::BoxesNMS(Bboxes[0], 0.5);
    Boxes::SmoothBoxes(currBoxes, Bboxes[0], 0.8);
    currBoxes = Bboxes[0];
    Image::ImRectangle(im_mat, Bboxes[0], true);
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

void Yolo::PredictYoloDetections(std::vector<image> &images,
                                 std::vector<VecBox> &Bboxes) {
  size_t num_im = images.size();

  int batch = net_.batch_;
  size_t batch_off = num_im % batch;
  size_t batch_num = num_im / batch + (batch_off > 0 ? 1 : 0);

  image im_res = Image::MakeImage(3, net_.in_h_, net_.in_w_);
  float *batch_data = new float[batch * net_.in_num_];
  float *predictions = NULL;
  for (int count = 0, b = 1; b <= batch_num; ++b) {
    float *index = batch_data;
    int c = 0;
    for (int i = count; i < b * batch && i < num_im; ++i, ++c) {
      Image::ImResize(images[i], im_res);
      Image::GenBatchData(im_res, index);
      index += net_.in_num_;
    }
    predictions = net_.PredictNetwork(batch_data);
    index = predictions;
    for (int i = 0; i < c; ++i) {
      VecBox boxes(grid_size_ * grid_size_ * box_num_);
      ConvertYoloDetections(index, class_num_, box_num_, sqrt_box_, grid_size_,
                            images[count + i].w, images[count + i].h, boxes);
      Bboxes.push_back(boxes);
      index += out_num_;
    }
    count += c;
  }
  delete batch_data;
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

      x = constrain(0, 1, x - w / 2);
      y = constrain(0, 1, y - h / 2);
      w = constrain(0, 1, x + w) - x;
      h = constrain(0, 1, y + h) - y;

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

void Yolo::PrintYoloDetections(std::ofstream &file, VecBox &boxes, int count) {
  for (int b = 0; b < boxes.size(); ++b) {
    Box box = boxes[b];
    if (box.class_index == -1)
      continue;
    file << count << ", " << box.x << ", " << box.y << ", " << box.w << ", "
         << box.h << ", " << box.score << endl;
  }
}
