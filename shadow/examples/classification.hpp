#ifndef SHADOW_EXAMPLES_CLASSIFICATION_HPP
#define SHADOW_EXAMPLES_CLASSIFICATION_HPP

#include "method.hpp"

namespace Shadow {

class Classification final : public Method {
 public:
  Classification() = default;
  ~Classification() override { Release(); }

  void Setup(const std::string &model_file, const VecInt &classes,
             int batch) override;

  void Predict(const JImage &im_src, const VecRectF &rois,
               std::vector<std::map<std::string, VecFloat>> *scores) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<std::map<std::string, VecFloat>> *scores) override;
#endif

  void Release() override;

 private:
  void Process(const float *data,
               std::vector<std::map<std::string, VecFloat>> *scores);

  Network net_;
  VecFloat in_data_;
  VecInt task_dims_;
  VecString task_names_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
  JImage im_ini_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_CLASSIFICATION_HPP
