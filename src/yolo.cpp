#include "yolo.hpp"
#include "shadow/kernel.hpp"
#include "shadow/util/jimage_proc.hpp"

using namespace std;

inline void GetBatchData(const JImage &im_src, float *batch_data) {
  if (im_src.data() == nullptr) Fatal("JImage data is NULL!");

  int c_ = im_src.c_, h_ = im_src.h_, w_ = im_src.w_;
  Order order_ = im_src.order();
  const unsigned char *data_src = im_src.data();

  bool is_rgb = false;
  if (order_ == kRGB) {
    is_rgb = true;
  } else if (order_ == kBGR) {
    is_rgb = false;
  } else {
    Fatal("Unsupported format to get batch data!");
  }
  int ch_src, offset, count = 0, step = w_ * c_;
  for (int c = 0; c < c_; ++c) {
    ch_src = is_rgb ? c : c_ - c - 1;
    for (int h = 0; h < h_; ++h) {
      for (int w = 0; w < w_; ++w) {
        offset = h * step + w * c_;
        batch_data[count++] = data_src[offset + ch_src];
      }
    }
  }
}

Yolo::Yolo(string cfg_file, string weight_file, float threshold) {
  cfg_file_ = cfg_file;
  weight_file_ = weight_file;
  threshold_ = threshold;
}

Yolo::~Yolo() {}

void Yolo::Setup(int batch, VecRectF *rois) {
#if defined(USE_CUDA) | defined(USE_CL)
  Kernel::Setup();
#endif

  net_.LoadModel(cfg_file_, weight_file_, batch);

  batch_ = net_.in_shape_.dim(0);
  in_num_ =
      net_.in_shape_.dim(1) * net_.in_shape_.dim(2) * net_.in_shape_.dim(3);
  const Layer *layer = net_.GetLayerByName("yolo_output");
  out_num_ = layer->layer_param_.connected_param().num_output();

  class_num_ = 2;
  batch_data_ = new float[batch_ * in_num_];
  predictions_ = new float[batch_ * out_num_];
  im_ini_ = new JImage();
  im_res_ = new JImage(3, net_.in_shape_.dim(2), net_.in_shape_.dim(3));
  if (rois == nullptr) {
    rois_.push_back(RectF(0.f, 0.f, 1.f, 1.f));
  } else {
    rois_ = *rois;
  }
}

void Yolo::Release() {
  net_.Release();
#if defined(USE_CUDA) | defined(USE_CL)
  Kernel::Release();
#endif
}

void Yolo::Test(string image_file) {
  Timer timer;
  timer.start();
  im_ini_->Read(image_file);
  vector<VecBox> Bboxes;
  PredictYoloDetections(im_ini_, &Bboxes);
  Boxes::AmendBoxes(&Bboxes, rois_, im_ini_->h_, im_ini_->w_);
  VecBox boxes = Boxes::BoxesNMS(Bboxes, 0.5);
  cout << "Predicted in " << timer.get_second() << " seconds" << endl;

  for (int i = 0; i < boxes.size(); ++i) {
    Box box = boxes[i];
    Scalar scalar;
    if (box.class_index == 0)
      scalar = Scalar(0, 255, 0);
    else
      scalar = Scalar(0, 0, 255);
    JImageProc::Rectangle(im_ini_, box.RectInt(), scalar);
  }
  im_ini_->Show("result");
}

void Yolo::BatchTest(string list_file, bool image_write) {
  vector<string> image_list = Util::load_list(list_file);
  size_t num_im = image_list.size();

  Timer timer;
  timer.start();
  ofstream file(Util::find_replace_last(list_file, ".", "-result."));
  for (int i = 0; i < num_im; ++i) {
    im_ini_->Read(image_list[i]);
    vector<VecBox> Bboxes;
    PredictYoloDetections(im_ini_, &Bboxes);
    Boxes::AmendBoxes(&Bboxes, rois_, im_ini_->h_, im_ini_->w_);
    VecBox boxes = Boxes::BoxesNMS(Bboxes, 0.5);
    if (image_write) {
      string path = Util::find_replace_last(image_list[i], ".", "-result.");
      for (int j = 0; j < boxes.size(); ++j) {
        Box box = boxes[j];
        Scalar scalar;
        if (box.class_index == 0)
          scalar = Scalar(0, 255, 0);
        else
          scalar = Scalar(0, 0, 255);
        JImageProc::Rectangle(im_ini_, box.RectInt(), scalar);
      }
      im_ini_->Write(path);
    }
    PrintYoloDetections(boxes, i + 1, &file);
    if (!((i + 1) % 100))
      cout << "Processing: " << i + 1 << " / " << num_im << endl;
  }
  file.close();
  cout << "Processing: " << num_im << " / " << num_im << endl;
  double sec = timer.get_second();
  cout << "Processed in: " << sec << " seconds, each frame: " << sec / num_im
       << "seconds" << endl;
}

#if defined(USE_OpenCV)
void Yolo::VideoTest(string video_file, bool video_show, bool video_write) {
  cv::VideoCapture capture;
  if (!capture.open(video_file))
    Fatal("error when opening video file " + video_file);
  float rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer;
  if (video_write) {
    string outfile = Util::change_extension(video_file, "-result.avi");
    int format = CV_FOURCC('H', '2', '6', '4');
    writer.open(outfile, format, rate, cv::Size(width, height));
  }
  CaptureTest(capture, "yolo video test!-:)", video_show, writer, video_write);
  capture.release();
  writer.release();
}

void Yolo::Demo(int camera, bool video_write) {
  cv::VideoCapture capture;
  if (!capture.open(camera)) Fatal("error when opening camera!");
  float rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter writer;
  if (video_write) {
    int format = CV_FOURCC('H', '2', '6', '4');
    writer.open("./data/demo-result.avi", format, rate,
                cv::Size(width, height));
  }
  CaptureTest(capture, "yolo demo!-:)", true, writer, video_write);
  capture.release();
  writer.release();
}

void Yolo::CaptureTest(cv::VideoCapture capture, string window_name,
                       bool video_show, cv::VideoWriter writer,
                       bool video_write) {
  float rate = static_cast<float>(capture.get(CV_CAP_PROP_FPS));
  if (video_show) cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::Mat im_mat;

  vector<VecBox> Bboxes;
  VecBox boxes, oldBoxes;
  Timer timer;
  while (capture.read(im_mat)) {
    if (im_mat.empty()) break;
    timer.start();
    im_ini_->FromMat(im_mat);
    PredictYoloDetections(im_ini_, &Bboxes);
    Boxes::AmendBoxes(&Bboxes, rois_, im_ini_->h_, im_ini_->w_);
    boxes = Boxes::BoxesNMS(Bboxes, 0.5);
    Boxes::SmoothBoxes(oldBoxes, &boxes, 0.3);
    oldBoxes = boxes;
    DrawYoloDetections(boxes, &im_mat, true);
    if (video_write) writer.write(im_mat);
    double sec = timer.get_second();
    int waittime = static_cast<int>(1000.0f * (1.0f / rate - sec));
    waittime = waittime > 1 ? waittime : 1;
    if (video_show) {
      cv::imshow(window_name, im_mat);
      if ((cv::waitKey(waittime) % 256) == 27) break;
      cout << "FPS:" << 1.0 / sec << endl;
    }
  }
}
#endif

void Yolo::PredictYoloDetections(JImage *image, vector<VecBox> *Bboxes) {
  Bboxes->clear();
  size_t num_im = rois_.size();
  size_t batch_off = num_im % batch_;
  size_t batch_num = num_im / batch_ + (batch_off > 0 ? 1 : 0);

  for (int count = 0, b = 1; b <= batch_num; ++b) {
    int c = 0;
    for (int i = count; i < b * batch_ && i < num_im; ++i, ++c) {
      JImageProc::CropResize(*image, im_res_, rois_[i], net_.in_shape_.dim(2),
                             net_.in_shape_.dim(3));
      GetBatchData(*im_res_, batch_data_);
    }
    net_.Forward(batch_data_);
    const Layer *layer = net_.GetLayerByName("yolo_output");
    if (layer != nullptr)
      layer->top_[0]->copy_data(predictions_);
    else
      Fatal("Unknown Blob name yolo_output");
    for (int i = 0; i < c; ++i) {
      VecBox boxes(98);
      int height = static_cast<int>(rois_[count + i].h * image->h_);
      int width = static_cast<int>(rois_[count + i].w * image->w_);
      ConvertYoloDetections(predictions_ + i * out_num_, class_num_, width,
                            height, &boxes);
      Bboxes->push_back(boxes);
    }
    count += c;
  }
}

void Yolo::ConvertYoloDetections(float *predictions, int classes, int width,
                                 int height, VecBox *boxes) {
  int side = 7, box_num = 2;
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
      w = pow(predictions[box_index + 2], 2);
      h = pow(predictions[box_index + 3], 2);

      x = x - w / 2;
      y = y - h / 2;

      (*boxes)[index].x = Util::constrain(0.f, width - 1.f, x * width);
      (*boxes)[index].y = Util::constrain(0.f, height - 1.f, y * height);
      (*boxes)[index].w = Util::constrain(0.f, width - 1.f, w * width);
      (*boxes)[index].h = Util::constrain(0.f, height - 1.f, h * height);

      float max_score = 0;
      int max_index = -1;
      for (int j = 0; j < classes; ++j) {
        float score = scale * predictions[i * classes + j];
        if (score > threshold_ && score > max_score) {
          max_score = score;
          max_index = j;
        }
      }
      (*boxes)[index].score = max_score;
      (*boxes)[index].class_index = max_index;
    }
  }
}

#if defined(USE_OpenCV)
void Yolo::DrawYoloDetections(const VecBox &boxes, cv::Mat *im_mat,
                              bool console_show) {
  for (int b = 0; b < boxes.size(); ++b) {
    int classindex = boxes[b].class_index;
    if (classindex == -1) continue;

    cv::Scalar scalar;
    if (classindex == 0)
      scalar = cv::Scalar(0, 255, 0);
    else
      scalar = cv::Scalar(255, 0, 0);

    Box box = boxes[b];
    cv::rectangle(*im_mat,
                  cv::Point(static_cast<int>(box.x), static_cast<int>(box.y)),
                  cv::Point(static_cast<int>(box.x + box.w),
                            static_cast<int>(box.y + box.h)),
                  scalar, 2, 8, 0);
    if (console_show) {
      cout << "x = " << box.x << ", y = " << box.y << ", w = " << box.w
           << ", h = " << box.h << ", score = " << box.score
           << ", label = " << box.class_index << endl;
    }
  }
}
#endif

void Yolo::PrintYoloDetections(const VecBox &boxes, int count, ofstream *file) {
  for (int b = 0; b < boxes.size(); ++b) {
    Box box = boxes[b];
    if (box.class_index == -1) continue;
    *file << count << ", " << box.x << ", " << box.y << ", " << box.w << ", "
          << box.h << ", " << box.score << endl;
  }
}
