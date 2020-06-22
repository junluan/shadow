#ifndef SHADOW_OPERATORS_DECODE_BOX_OP_HPP
#define SHADOW_OPERATORS_DECODE_BOX_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DecodeBoxOp : public Operator {
 public:
  DecodeBoxOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    method_ = get_single_argument<int>("method", 0);
    num_classes_ = get_single_argument<int>("num_classes", 1);
    CHECK_GT(num_classes_, 1);
    output_max_score_ = get_single_argument<bool>("output_max_score", true);
    background_label_id_ = get_single_argument<int>("background_label_id", 0);
    objectness_score_ = get_single_argument<float>("objectness_score", 0.01f);
    masks_ = get_repeated_argument<int>("masks");
  }

  void Forward() override;

 private:
  enum { kSSD = 0, kRefineDet = 1, kYoloV3 = 2 };

  int method_, num_classes_, background_label_id_;
  float objectness_score_;
  bool output_max_score_;
  VecInt masks_;
};

namespace Vision {

template <typename T>
void DecodeSSDBoxes(const T *mbox_loc, const T *mbox_conf,
                    const T *mbox_priorbox, int batch, int num_priors,
                    int num_classes, bool output_max_score, T *decode_box,
                    Context *context);

template <typename T>
void DecodeRefineDetBoxes(const T *odm_loc, const T *odm_conf,
                          const T *arm_priorbox, const T *arm_conf,
                          const T *arm_loc, int batch, int num_priors,
                          int num_classes, int background_label_id,
                          float objectness_score, bool output_max_score,
                          T *decode_box, Context *context);

template <typename T>
void DecodeYoloV3Boxes(const T *in_data, const T *biases, int batch,
                       int num_priors, int out_h, int out_w, int mask,
                       int num_classes, bool output_max_score, T *decode_box,
                       Context *context);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DECODE_BOX_OP_HPP
