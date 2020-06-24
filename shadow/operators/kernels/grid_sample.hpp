#ifndef SHADOW_OPERATORS_KERNELS_GRID_SAMPLE_HPP_
#define SHADOW_OPERATORS_KERNELS_GRID_SAMPLE_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void GridSample(const T* in_data, const VecInt& in_shape,
                const float* grid_data, int mode, int padding_mode,
                const VecInt& out_shape, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class GridSampleKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& grid,
                   std::shared_ptr<Blob>& output, Workspace* ws, int mode,
                   int padding_mode) = 0;
};

template <DeviceType D>
class GridSampleKernelDefault : public GridSampleKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& grid, std::shared_ptr<Blob>& output,
           Workspace* ws, int mode, int padding_mode) override {
    Vision::GridSample<D, float>(input->data<float>(), input->shape(),
                                 grid->data<float>(), mode, padding_mode,
                                 output->shape(), output->mutable_data<float>(),
                                 ws->Ctx());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_GRID_SAMPLE_HPP_
