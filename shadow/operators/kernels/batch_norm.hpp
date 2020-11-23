#ifndef SHADOW_OPERATORS_KERNELS_BATCH_NORM_HPP_
#define SHADOW_OPERATORS_KERNELS_BATCH_NORM_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void BatchNorm(const float* in_data, int count, const float* mean_data,
               const float* variance_data, int channel, int inner_num,
               float scale_factor, float eps, float* out_data,
               Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class BatchNormKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& mean,
                   const std::shared_ptr<Blob>& variance,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   float scale_factor, float eps) = 0;
};

template <DeviceType D>
class BatchNormKernelDefault : public BatchNormKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& mean,
           const std::shared_ptr<Blob>& variance, std::shared_ptr<Blob>& output,
           Workspace* ws, float scale_factor, float eps) override {
    int channel = input->shape(1);

    CHECK_EQ(channel, mean->count());
    CHECK_EQ(channel, variance->count());

    Vision::BatchNorm<D, float>(input->data<float>(), input->count(),
                                mean->data<float>(), variance->data<float>(),
                                channel, input->count(2), scale_factor, eps,
                                output->mutable_data<float>(), ws->Ctx().get());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_BATCH_NORM_HPP_
