#ifndef SHADOW_OPERATORS_KERNELS_SSD_NORMALIZE_HPP_
#define SHADOW_OPERATORS_KERNELS_SSD_NORMALIZE_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

#include "scale.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void SSDNormalize(const T* in_data, int outer_num, int channel, int inner_num,
                  float eps, T* val_data, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class SSDNormalizeKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& scale,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   bool across_spatial, bool channel_shared, float eps) = 0;
};

template <DeviceType D>
class SSDNormalizeKernelDefault : public SSDNormalizeKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& scale, std::shared_ptr<Blob>& output,
           Workspace* ws, bool across_spatial, bool channel_shared,
           float eps) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int batch = input->shape(0), channel = input->shape(1),
        spatial_dim = input->count(2);
    int count = input->count(), num = input->num();

    Blas::Square<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());

    if (across_spatial) {
      for (int b = 0; b < batch; ++b) {
        int offset = b * num;
        float sum = 0;
        Blas::BlasSasum<D, float>(num, out_data, offset, &sum, ws->Ctx());
        Blas::Mul<D, float>(num, in_data, offset, 1.f / std::sqrt(sum + eps),
                            out_data, offset, ws->Ctx());
      }
    } else {
      ws->GrowTempBuffer(batch * spatial_dim * sizeof(float));

      auto scalar = ws->CreateTempBlob({batch, spatial_dim}, DataType::kF32);

      Vision::SSDNormalize<D, float>(in_data, batch, channel, spatial_dim, eps,
                                     scalar->mutable_data<float>(), out_data,
                                     ws->Ctx());
    }

    if (scale != nullptr) {
      if (channel_shared) {
        CHECK_EQ(scale->count(), 1);
        float scale_data = 1;
        scale->get_data<float>(&scale_data, 1);
        Blas::BlasSscal<D, float>(count, scale_data, out_data, 0, ws->Ctx());
      } else {
        Vision::Scale<D, float>(out_data, count, scale->data<float>(), channel,
                                spatial_dim, out_data, ws->Ctx());
      }
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_SSD_NORMALIZE_HPP_
