#ifndef SHADOW_OPERATORS_KERNELS_SCALE_HPP_
#define SHADOW_OPERATORS_KERNELS_SCALE_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void ScaleBias(const T* in_data, int count, const T* scale_data,
               const T* bias_data, int scale_num, int inner_num, T* out_data,
               Context* context);

template <DeviceType D, typename T>
void Scale(const T* in_data, int count, const T* scale_data, int scale_num,
           int inner_num, T* out_data, Context* context);

template <DeviceType D, typename T>
void Bias(const T* in_data, int count, const T* bias_data, int scale_num,
          int inner_num, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class ScaleKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& scale,
                   const std::shared_ptr<Blob>& bias,
                   std::shared_ptr<Blob>& output, Workspace* ws, int axis) = 0;

  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int axis,
                   const VecFloat& scale_value, const VecFloat& bias_value) = 0;
};

template <DeviceType D>
class ScaleKernelDefault : public ScaleKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& scale,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int axis) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int count = input->count();

    CHECK(scale != nullptr || bias != nullptr);

    if (scale != nullptr && bias != nullptr) {
      CHECK(scale->shape() == bias->shape());
      int inner_dim = input->count(axis + scale->num_axes());
      Vision::ScaleBias<D, float>(in_data, count, scale->data<float>(),
                                  bias->data<float>(), scale->count(),
                                  inner_dim, out_data, ws->Ctx().get());
    } else if (scale != nullptr) {
      int inner_dim = input->count(axis + scale->num_axes());
      Vision::Scale<D, float>(in_data, count, scale->data<float>(),
                              scale->count(), inner_dim, out_data,
                              ws->Ctx().get());
    } else {
      int inner_dim = input->count(axis + bias->num_axes());
      Vision::Bias<D, float>(in_data, count, bias->data<float>(), bias->count(),
                             inner_dim, out_data, ws->Ctx().get());
    }
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int axis, const VecFloat& scale_value,
           const VecFloat& bias_value) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int dim = input->shape(axis), inner_dim = input->count(axis + 1),
        count = input->count();

    CHECK(!scale_value.empty() || !bias_value.empty());

    if (!scale_value.empty() && !bias_value.empty()) {
      CHECK_EQ(scale_value.size(), dim);
      CHECK_EQ(bias_value.size(), dim);
      ws->GrowTempBuffer(2 * dim * sizeof(float));
      auto scale = ws->CreateTempBlob({dim}, DataType::kF32);
      auto bias = ws->CreateTempBlob({dim}, DataType::kF32);
      scale->set_data<float>(scale_value.data(), dim);
      bias->set_data<float>(bias_value.data(), dim);
      Vision::ScaleBias<D, float>(in_data, count, scale->data<float>(),
                                  bias->data<float>(), dim, inner_dim, out_data,
                                  ws->Ctx().get());
    } else if (!scale_value.empty()) {
      CHECK_EQ(scale_value.size(), dim);
      ws->GrowTempBuffer(dim * sizeof(float));
      auto scale = ws->CreateTempBlob({dim}, DataType::kF32);
      scale->set_data<float>(scale_value.data(), dim);
      Vision::Scale<D, float>(in_data, count, scale->data<float>(), dim,
                              inner_dim, out_data, ws->Ctx().get());
    } else {
      CHECK_EQ(bias_value.size(), dim);
      ws->GrowTempBuffer(dim * sizeof(float));
      auto bias = ws->CreateTempBlob({dim}, DataType::kF32);
      bias->set_data<float>(bias_value.data(), dim);
      Vision::Bias<D, float>(in_data, count, bias->data<float>(), dim,
                             inner_dim, out_data, ws->Ctx().get());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_SCALE_HPP_
