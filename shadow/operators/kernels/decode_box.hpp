#ifndef SHADOW_OPERATORS_KERNELS_DECODE_BOX_HPP_
#define SHADOW_OPERATORS_KERNELS_DECODE_BOX_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void DecodeSSDBoxes(const T* mbox_loc, const T* mbox_conf,
                    const T* mbox_priorbox, int batch, int num_priors,
                    int num_classes, bool output_max_score, T* decode_box,
                    Context* context);

template <DeviceType D, typename T>
void DecodeRefineDetBoxes(const T* odm_loc, const T* odm_conf,
                          const T* arm_priorbox, const T* arm_conf,
                          const T* arm_loc, int batch, int num_priors,
                          int num_classes, int background_label_id,
                          float objectness_score, bool output_max_score,
                          T* decode_box, Context* context);

template <DeviceType D, typename T>
void DecodeYoloV3Boxes(const T* in_data, const T* biases, int batch,
                       int num_priors, int out_h, int out_w, int mask,
                       int num_classes, bool output_max_score, T* decode_box,
                       Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class DecodeBoxKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& mbox_loc,
                   const std::shared_ptr<Blob>& mbox_conf,
                   const std::shared_ptr<Blob>& mbox_priorbox,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   int num_classes, bool output_max_score) = 0;

  virtual void Run(const std::shared_ptr<Blob>& odm_loc,
                   const std::shared_ptr<Blob>& odm_conf,
                   const std::shared_ptr<Blob>& arm_priorbox,
                   const std::shared_ptr<Blob>& arm_conf,
                   const std::shared_ptr<Blob>& arm_loc,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   int num_classes, int background_label_id,
                   float objectness_score, bool output_max_score) = 0;

  virtual void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
                   const std::shared_ptr<Blob>& biases,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   int num_classes, bool output_max_score,
                   const VecInt& masks) = 0;
};

template <DeviceType D>
class DecodeBoxKernelDefault : public DecodeBoxKernel {
 public:
  void Run(const std::shared_ptr<Blob>& mbox_loc,
           const std::shared_ptr<Blob>& mbox_conf,
           const std::shared_ptr<Blob>& mbox_priorbox,
           std::shared_ptr<Blob>& output, Workspace* ws, int num_classes,
           bool output_max_score) override {
    Vision::DecodeSSDBoxes<D, float>(
        mbox_loc->data<float>(), mbox_conf->data<float>(),
        mbox_priorbox->data<float>(), output->shape(0), output->shape(1),
        num_classes, output_max_score, output->mutable_data<float>(),
        ws->Ctx());
  }

  void Run(const std::shared_ptr<Blob>& odm_loc,
           const std::shared_ptr<Blob>& odm_conf,
           const std::shared_ptr<Blob>& arm_priorbox,
           const std::shared_ptr<Blob>& arm_conf,
           const std::shared_ptr<Blob>& arm_loc, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_classes, int background_label_id,
           float objectness_score, bool output_max_score) override {
    Vision::DecodeRefineDetBoxes<D, float>(
        odm_loc->data<float>(), odm_conf->data<float>(),
        arm_priorbox->data<float>(), arm_conf->data<float>(),
        arm_loc->data<float>(), output->shape(0), output->shape(1), num_classes,
        background_label_id, objectness_score, output_max_score,
        output->mutable_data<float>(), ws->Ctx());
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           const std::shared_ptr<Blob>& biases, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_classes, bool output_max_score,
           const VecInt& masks) override {
    const auto* biases_data = biases->data<float>();
    auto* out_data = output->mutable_data<float>();

    int batch = output->shape(0), num_priors = output->shape(1),
        out_stride = output->shape(2);

    for (int n = 0; n < masks.size(); ++n) {
      const auto& input = inputs[n];
      int mask = masks[n], out_h = input->shape(1), out_w = input->shape(2);
      Vision::DecodeYoloV3Boxes<D, float>(
          input->data<float>(), biases_data, batch, num_priors, out_h, out_w,
          mask, num_classes, output_max_score, out_data, ws->Ctx());
      biases_data += mask * 2;
      out_data += out_h * out_w * mask * out_stride;
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_DECODE_BOX_HPP_
