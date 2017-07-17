#include "classification.hpp"

namespace Shadow {

void Classification::Setup(const std::string &model_file, const VecInt &classes,
                           int batch) {
  net_.Setup();

  net_.LoadModel(model_file, batch);

  batch_ = net_.in_shape()[0];
  in_c_ = net_.in_shape()[1];
  in_h_ = net_.in_shape()[2];
  in_w_ = net_.in_shape()[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);

  task_names_ = VecString{"score"};
  task_dims_ = classes;
  CHECK_EQ(task_names_.size(), task_dims_.size());
  int num_dim = 0;
  for (const auto dim : task_dims_) {
    num_dim += dim;
  }
  CHECK_EQ(num_dim, net_.GetBlobByName("softmax")->num());
}

void Classification::Predict(
    const JImage &im_src, const VecRectF &rois,
    std::vector<std::map<std::string, VecFloat>> *scores) {
  CHECK_LE(rois.size(), batch_);
  for (int b = 0; b < rois.size(); ++b) {
    ConvertData(im_src, in_data_.data() + b * in_num_, rois[b], in_c_, in_h_,
                in_w_);
  }

  Process(in_data_.data(), scores);

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

void Classification::Release() {
  net_.Release();

  in_data_.clear();
}

void Classification::Process(
    const float *data, std::vector<std::map<std::string, VecFloat>> *scores) {
  net_.Forward(data);

  const float *softmax_data = net_.GetBlobDataByName("softmax");

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
