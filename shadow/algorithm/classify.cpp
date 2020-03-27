#include "classify.hpp"

#include "util/io.hpp"

namespace Shadow {

void Classify::Setup(const std::string &model_file) {
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
  prob_str_ = out_blob[0];

  const auto &data_shape = net_.GetBlobShapeByName<float>(in_str_);
  CHECK_EQ(data_shape.size(), 4);

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  num_classes_ = net_.get_single_argument<int>("num_classes", 1000);
}

void Classify::Predict(const JImage &im_src, const RectF &roi,
                       std::map<std::string, VecFloat> *scores) {
  ConvertData(im_src, in_data_.data(), roi, in_c_, in_h_, in_w_);

  Process(in_data_, scores);
}

#if defined(USE_OpenCV)
void Classify::Predict(const cv::Mat &im_mat, const RectF &roi,
                       std::map<std::string, VecFloat> *scores) {
  ConvertData(im_mat, in_data_.data(), roi, in_c_, in_h_, in_w_);

  Process(in_data_, scores);
}
#endif

void Classify::Process(const VecFloat &in_data,
                       std::map<std::string, VecFloat> *scores) {
  std::map<std::string, void *> data_map;
  data_map[in_str_] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  const auto *prob_data = net_.GetBlobDataByName<float>(prob_str_);

  scores->clear();
  for (int b = 0; b < batch_; ++b) {
    int offset = b * num_classes_;
    (*scores)["score"] = VecFloat(prob_data + offset, prob_data + num_classes_);
  }
}

}  // namespace Shadow
