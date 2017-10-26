#ifndef SHADOW_EXAMPLES_DETECTION_MTCNN_HPP
#define SHADOW_EXAMPLES_DETECTION_MTCNN_HPP

#include "method.hpp"

namespace Shadow {

struct BoxInfo {
  BoxF box;
  float box_reg[4], landmark[10];
};

using VecBoxInfo = std::vector<BoxInfo>;

class DetectionMTCNN final : public Method {
 public:
  DetectionMTCNN() = default;
  ~DetectionMTCNN() override { Release(); }

  void Setup(const VecString &model_files, const VecInt &in_shape) override;

  void Predict(const JImage &im_src, const VecRectF &rois,
               std::vector<VecBoxF> *Gboxes,
               std::vector<std::vector<VecPointF>> *Gpoints) override;
#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<VecBoxF> *Gboxes,
               std::vector<std::vector<VecPointF>> *Gpoints) override;
#endif

  void Release() override;

 private:
  void Process_net_p(const float *data, const VecInt &in_shape, float threshold,
                     float scale, VecBoxInfo *boxes);
  void Process_net_r(const float *data, const VecInt &in_shape, float threshold,
                     const VecBoxInfo &net_12_boxes, VecBoxInfo *boxes);
  void Process_net_o(const float *data, const VecInt &in_shape, float threshold,
                     const VecBoxInfo &net_24_boxes, VecBoxInfo *boxes);

  void CalculateScales(float height, float width, float factor, float max_side,
                       float min_side, VecFloat *scales);

  void BoxRegression(VecBoxInfo &boxes);

  void Box2SquareWithConstrain(VecBoxInfo &boxes, float height, float width);
  void BoxWithConstrain(VecBoxInfo &boxes, float height, float width);

  Network net_p_, net_r_, net_o_;
  VecFloat net_p_in_data_, net_r_in_data_, net_o_in_data_, thresholds_, scales_;
  VecInt net_p_in_shape_, net_r_in_shape_, net_o_in_shape_;
  VecBoxInfo net_p_boxes_, net_r_boxes_, net_o_boxes_;
  int net_p_stride_, net_p_cell_size_;
  int net_r_in_c_, net_r_in_h_, net_r_in_w_, net_r_in_num_;
  int net_o_in_c_, net_o_in_h_, net_o_in_w_, net_o_in_num_;
  float factor_, max_side_, min_side_;
  JImage im_ini_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DETECTION_MTCNN_HPP
