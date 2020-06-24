#ifndef SHADOW_OPERATORS_KERNELS_GROUP_NORM_HPP_
#define SHADOW_OPERATORS_KERNELS_GROUP_NORM_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

#include "scale.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void SubtractMeanAndSquare(const T* in_data, const T* mean_data, int count,
                           int inner_num, T* out_data, T* square_data,
                           Context* context);

template <DeviceType D, typename T>
void DivideVariance(const T* in_data, const T* variance_data, int count,
                    int inner_num, float eps, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class GroupNormKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& scale,
                   const std::shared_ptr<Blob>& bias,
                   std::shared_ptr<Blob>& output, Workspace* ws, int group,
                   float eps) = 0;
};

template <DeviceType D>
class GroupNormKernelDefault : public GroupNormKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& scale,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int group, float eps) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int count = input->count();
    int outer_num = input->shape(0) * group, inner_num = count / outer_num;

    ws->GrowTempBuffer((2 * outer_num + count + inner_num) * sizeof(float));

    auto mean = ws->CreateTempBlob({outer_num}, DataType::kF32);
    auto variance = ws->CreateTempBlob({outer_num}, DataType::kF32);
    auto temp = ws->CreateTempBlob(input->shape(), DataType::kF32);
    auto sum_inner_multiplier = ws->CreateTempBlob({inner_num}, DataType::kF32);

    Blas::Set<D, float>(inner_num, 1,
                        sum_inner_multiplier->mutable_data<float>(), 0,
                        ws->Ctx());

    Blas::BlasSgemv<D, float>(0, outer_num, inner_num, 1.f / inner_num, in_data,
                              0, sum_inner_multiplier->data<float>(), 0, 0,
                              mean->mutable_data<float>(), 0, ws->Ctx());

    Vision::SubtractMeanAndSquare<D, float>(
        in_data, mean->data<float>(), count, inner_num, out_data,
        temp->mutable_data<float>(), ws->Ctx());

    Blas::BlasSgemv<D, float>(0, outer_num, inner_num, 1.f / inner_num,
                              temp->data<float>(), 0,
                              sum_inner_multiplier->data<float>(), 0, 0,
                              variance->mutable_data<float>(), 0, ws->Ctx());

    Vision::DivideVariance<D, float>(out_data, variance->data<float>(), count,
                                     inner_num, eps, out_data, ws->Ctx());

    if (scale != nullptr && bias != nullptr) {
      int channel = input->shape(1), spatial_dim = input->count(2);
      CHECK_EQ(scale->count(), channel);
      CHECK_EQ(bias->count(), channel);
      Vision::ScaleBias<D, float>(out_data, count, scale->data<float>(),
                                  bias->data<float>(), channel, spatial_dim,
                                  out_data, ws->Ctx());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_GROUP_NORM_HPP_
