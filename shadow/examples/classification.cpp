#include "classification.hpp"

namespace Shadow {

void Classification::Setup(const VecString &model_files,
                           const VecInt &in_shape) {
  net_.Setup();

  net_.LoadModel(model_files[0]);

  const auto &in_blob = net_.in_blob();
  CHECK_EQ(in_blob.size(), 1);
  in_str_ = in_blob[0];

  const auto &out_blob = net_.out_blob();
  CHECK_EQ(out_blob.size(), 1);
  prob_str_ = out_blob[0];

  auto data_shape = net_.GetBlobByName<float>(in_str_)->shape();
  CHECK_EQ(data_shape.size(), 4);
  CHECK_EQ(in_shape.size(), 1);
  if (data_shape[0] != in_shape[0]) {
    data_shape[0] = in_shape[0];
    std::map<std::string, VecInt> shape_map;
    shape_map[in_str_] = data_shape;
    net_.Reshape(shape_map);
  }

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  task_names_ = VecString{"score"};
  task_dims_ = net_.num_class();
  CHECK_EQ(task_names_.size(), task_dims_.size());
  int num_dim = 0;
  for (const auto dim : task_dims_) {
    num_dim += dim;
  }
  CHECK_EQ(num_dim, net_.GetBlobByName<float>(prob_str_)->num());
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
  CHECK_LE(rois.size(), batch_);
  for (int b = 0; b < rois.size(); ++b) {
    ConvertData(im_mat, in_data_.data() + b * in_num_, rois[b], in_c_, in_h_,
                in_w_);
  }

  Process(in_data_, scores);

  CHECK_EQ(scores->size(), rois.size());
}
#endif

void Classification::Release() { net_.Release(); }

void Classification::Process(
    const VecFloat &in_data,
    std::vector<std::map<std::string, VecFloat>> *scores) {
  std::map<std::string, float *> data_map;
  data_map[in_str_] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  const auto *softmax_data = net_.GetBlobDataByName<float>(prob_str_);

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
