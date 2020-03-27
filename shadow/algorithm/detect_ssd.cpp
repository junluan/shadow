#include "detect_ssd.hpp"

#include "util/io.hpp"

namespace Shadow {

void DetectSSD::Setup(const std::string &model_file) {
#if defined(USE_Protobuf)
  shadow::MetaNetParam meta_net_param;
  CHECK(IO::ReadProtoFromBinaryFile(model_file, &meta_net_param))
      << "Error when loading proto binary file: " << model_file;

  ArgumentHelper arguments;
  arguments.AddSingleArgument<std::string>("backend_type", "Native");

  net_.LoadXModel(meta_net_param.network(0), arguments);

#else
  LOG(FATAL) << "Unsupported load binary model, recompiled with USE_Protobuf";
#endif

  const auto &in_blob = net_.in_blob();
  CHECK_EQ(in_blob.size(), 1);
  in_str_ = in_blob[0];

  const auto &out_blob = net_.out_blob();
  CHECK_EQ(out_blob.size(), 1);
  out_str_ = out_blob[0];

  const auto &data_shape = net_.GetBlobShapeByName<float>(in_str_);
  CHECK_EQ(data_shape.size(), 4);

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  threshold_ = net_.get_single_argument<float>("threshold", 0.6);
  background_label_id_ = 0;
  nms_threshold_ = net_.get_single_argument<float>("nms_threshold", 0.3);
}

void DetectSSD::Predict(const JImage &im_src, const RectF &roi, VecBoxF *boxes,
                        std::vector<VecPointF> *Gpoints) {
  ConvertData(im_src, in_data_.data(), roi, in_c_, in_h_, in_w_);

  std::vector<VecBoxF> Gboxes;
  Process(in_data_, &Gboxes);

  float height = roi.h, width = roi.w;
  for (auto &box : Gboxes[0]) {
    box.xmin *= width;
    box.xmax *= width;
    box.ymin *= height;
    box.ymax *= height;
  }

  *boxes = Gboxes[0];
}

#if defined(USE_OpenCV)
void DetectSSD::Predict(const cv::Mat &im_mat, const RectF &roi, VecBoxF *boxes,
                        std::vector<VecPointF> *Gpoints) {
  ConvertData(im_mat, in_data_.data(), roi, in_c_, in_h_, in_w_);

  std::vector<VecBoxF> Gboxes;
  Process(in_data_, &Gboxes);

  float height = roi.h, width = roi.w;
  for (auto &box : Gboxes[0]) {
    box.xmin *= width;
    box.xmax *= width;
    box.ymin *= height;
    box.ymax *= height;
  }

  *boxes = Gboxes[0];
}
#endif

void DetectSSD::Process(const VecFloat &in_data, std::vector<VecBoxF> *Gboxes) {
  std::map<std::string, void *> data_map;
  data_map[in_str_] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  const auto &out_shape = net_.GetBlobShapeByName<float>(out_str_);
  const auto *out_data = net_.GetBlobDataByName<float>(out_str_);

  int batch = out_shape[0], num_priors = out_shape[1], num_data = out_shape[2];

  Gboxes->clear();
  for (int b = 0; b < batch; ++b) {
    VecBoxF boxes;
    for (int n = 0; n < num_priors; ++n, out_data += num_data) {
      int label = static_cast<int>(out_data[0]);
      float score = out_data[1];
      if (label != background_label_id_ && score > threshold_) {
        BoxF box;
        box.xmin = out_data[2];
        box.ymin = out_data[3];
        box.xmax = out_data[4];
        box.ymax = out_data[5];
        box.score = score;
        box.label = label;
        boxes.push_back(box);
      }
    }
    Gboxes->push_back(Boxes::NMS(boxes, nms_threshold_));
  }
}

}  // namespace Shadow
