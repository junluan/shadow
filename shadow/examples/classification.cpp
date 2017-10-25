#include "classification.hpp"

namespace Shadow {

void Classification::Setup(const VecString &model_files, const VecInt &classes,
                           const VecInt &in_shape) {
  net_.Setup();

  net_.LoadModel(model_files[0]);

  auto data_shape = net_.GetBlobByName<float>("data")->shape();
  CHECK_EQ(data_shape.size(), 4);
  CHECK_EQ(in_shape.size(), 1);
  if (data_shape[0] != in_shape[0]) {
    data_shape[0] = in_shape[0];
    std::map<std::string, VecInt> shape_map;
    shape_map["data"] = data_shape;
    net_.Reshape(shape_map);
  }

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  task_names_ = VecString{"score"};
  task_dims_ = classes;
  CHECK_EQ(task_names_.size(), task_dims_.size());
  int num_dim = 0;
  for (const auto dim : task_dims_) {
    num_dim += dim;
  }
  CHECK_EQ(num_dim, net_.GetBlobByName<float>("softmax")->num());
}

void Classification::Predict(
    const JImage &im_src, const VecRectF &rois,
    std::vector<std::map<std::string, VecFloat>> *scores) {
  CHECK_LE(rois.size(), batch_);
  for (int b = 0; b < rois.size(); ++b) {
    ConvertData(im_src, in_data_.data() + b * in_num_, rois[b], in_c_, in_h_,
                in_w_);
  }

  Process(in_data_, scores);

  CHECK_EQ(scores->size(), rois.size());
}

#if defined(USE_OpenCV)
void Classification::Predict(
    const cv::Mat &im_mat, const VecRectF &rois,
    std::vector<std::map<std::string, VecFloat>> *scores) {
  im_ini_.FromMat(im_mat, true);
  Predict(im_ini_, rois, scores);
}
#endif

void Classification::Release() { net_.Release(); }

void Classification::Process(
    const VecFloat &in_data,
    std::vector<std::map<std::string, VecFloat>> *scores) {
  std::map<std::string, float *> data_map;
  data_map["data"] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  const auto *softmax_data = net_.GetBlobDataByName<float>("softmax");

  scores->clear();
  int offset = 0;
  for (int b = 0; b < batch_; ++b) {
    std::map<std::string, VecFloat> score_map;
    for (int n = 0; n < task_dims_.size(); ++n) {
      const auto &name = task_names_[n];
      int dim = task_dims_[n];
      VecFloat task_score(softmax_data + offset, softmax_data + offset + dim);
      score_map[name] = task_score;
      offset += dim;
    }
    scores->push_back(score_map);
  }
}

}  // namespace Shadow
