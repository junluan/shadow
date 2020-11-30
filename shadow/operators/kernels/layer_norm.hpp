#ifndef SHADOW_OPERATORS_KERNELS_LAYER_NORM_HPP_
#define SHADOW_OPERATORS_KERNELS_LAYER_NORM_HPP_

#include "group_norm.hpp"

namespace Shadow {

class LayerNormKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& scale,
                   const std::shared_ptr<Blob>& bias,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   const VecInt& normalized_shape, float eps) = 0;
};

template <DeviceType D>
class LayerNormKernelDefault : public LayerNormKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& scale,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, const VecInt& normalized_shape, float eps) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int inner_num = input->count(input->num_axes() - normalized_shape.size());
    int count = input->count(), outer_num = count / inner_num;

    ws->GrowTempBuffer((2 * outer_num + count + inner_num) * sizeof(float));

    auto mean = ws->CreateTempBlob({outer_num}, DataType::kF32);
    auto variance = ws->CreateTempBlob({outer_num}, DataType::kF32);
    auto temp = ws->CreateTempBlob(input->shape(), DataType::kF32);
    auto sum_inner_multiplier = ws->CreateTempBlob({inner_num}, DataType::kF32);

    Blas::Set<D, float>(inner_num, 1,
                        sum_inner_multiplier->mutable_data<float>(), 0,
                        ws->Ctx().get());

    Blas::BlasSgemv<D, float>(0, outer_num, inner_num, 1.f / inner_num, in_data,
                              0, sum_inner_multiplier->data<float>(), 0, 0,
                              mean->mutable_data<float>(), 0, ws->Ctx().get());

    Vision::SubtractMeanAndSquare<D, float>(
        in_data, mean->data<float>(), count, inner_num, out_data,
        temp->mutable_data<float>(), ws->Ctx().get());

    Blas::BlasSgemv<D, float>(
        0, outer_num, inner_num, 1.f / inner_num, temp->data<float>(), 0,
        sum_inner_multiplier->data<float>(), 0, 0,
        variance->mutable_data<float>(), 0, ws->Ctx().get());

    Vision::DivideVariance<D, float>(out_data, variance->data<float>(), count,
                                     inner_num, eps, out_data, ws->Ctx().get());

    if (scale != nullptr && bias != nullptr) {
      CHECK(scale->shape() == normalized_shape);
      CHECK(bias->shape() == normalized_shape);
      Vision::ScaleBias<D, float>(out_data, count, scale->data<float>(),
                                  bias->data<float>(), inner_num, 1, out_data,
                                  ws->Ctx().get());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_LAYER_NORM_HPP_
