#include "core/operator.hpp"

#include "kernels/decode_box.hpp"

namespace Shadow {

class DecodeBoxOp : public Operator {
 public:
  DecodeBoxOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    method_ = get_single_argument<int>("method", 0);
    num_classes_ = get_single_argument<int>("num_classes", 1);
    output_max_score_ = get_single_argument<bool>("output_max_score", true);
    background_label_id_ = get_single_argument<int>("background_label_id", 0);
    objectness_score_ = get_single_argument<float>("objectness_score", 0.01f);
    masks_ = get_repeated_argument<int>("masks");

    kernel_ = std::dynamic_pointer_cast<DecodeBoxKernel>(
        CreateKernel(op_param.type(), ws_->Ctx()->device_type()));
    CHECK_NOTNULL(kernel_);
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {
    auto& output = outputs[0];

    int out_stride = output_max_score_ ? 6 : (4 + num_classes_);

    if (method_ == kSSD) {
      CHECK_EQ(inputs.size(), 3);

      const auto& mbox_loc = inputs[0];
      const auto& mbox_conf = inputs[1];
      const auto& mbox_priorbox = inputs[2];

      int batch = mbox_loc->shape(0), num_priors = mbox_loc->shape(1) / 4;

      CHECK_EQ(mbox_conf->shape(1), num_priors * num_classes_);
      CHECK_EQ(mbox_priorbox->count(1), num_priors * 8);

      output->reshape({batch, num_priors, out_stride});

      kernel_->Run(mbox_loc, mbox_conf, mbox_priorbox, output, ws_,
                   num_classes_, output_max_score_);
    } else if (method_ == kRefineDet) {
      CHECK_EQ(inputs.size(), 5);

      const auto& odm_loc = inputs[0];
      const auto& odm_conf = inputs[1];
      const auto& arm_priorbox = inputs[2];
      const auto& arm_conf = inputs[3];
      const auto& arm_loc = inputs[4];

      int batch = odm_loc->shape(0), num_priors = odm_loc->shape(1) / 4;

      CHECK_EQ(odm_conf->shape(1), num_priors * num_classes_);
      CHECK_EQ(arm_priorbox->count(1), num_priors * 8);
      CHECK_EQ(arm_conf->shape(1), num_priors * 2);
      CHECK_EQ(arm_loc->shape(1), num_priors * 4);

      output->reshape({batch, num_priors, out_stride});

      kernel_->Run(odm_loc, odm_conf, arm_priorbox, arm_conf, arm_loc, output,
                   ws_, num_classes_, background_label_id_, objectness_score_,
                   output_max_score_);
    } else if (method_ == kYoloV3) {
      CHECK_EQ(inputs.size(), masks_.size() + 1);

      const auto& biases = inputs.back();

      CHECK_EQ(biases->count(),
               std::accumulate(masks_.begin(), masks_.end(), 0) * 2);

      int num_priors = 0;
      for (int n = 0; n < masks_.size(); ++n) {
        int mask = masks_[n];
        const auto& input = inputs[n];
        CHECK_EQ(input->shape(3), (4 + 1 + num_classes_) * mask);
        num_priors += input->count(1, 3) * mask;
      }

      output->reshape({inputs[0]->shape(0), num_priors, out_stride});

      kernel_->Run(
          std::vector<std::shared_ptr<Blob>>(inputs.begin(), inputs.end() - 1),
          biases, output, ws_, num_classes_, output_max_score_, masks_);
    } else {
      LOG(FATAL) << "Currently only support SSD, RefineDet or YoloV3";
    }
  }

 private:
  enum { kSSD = 0, kRefineDet = 1, kYoloV3 = 2 };

  int method_, num_classes_, background_label_id_;
  float objectness_score_;
  bool output_max_score_;
  VecInt masks_;

  std::shared_ptr<DecodeBoxKernel> kernel_ = nullptr;
};

REGISTER_OPERATOR(DecodeBox, DecodeBoxOp);

}  // namespace Shadow
