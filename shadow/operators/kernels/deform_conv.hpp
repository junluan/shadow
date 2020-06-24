#ifndef SHADOW_OPERATORS_KERNELS_DEFORM_CONV_HPP_
#define SHADOW_OPERATORS_KERNELS_DEFORM_CONV_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

#include "activate.hpp"
#include "scale.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void DeformIm2Col(const T* in_data, const VecInt& in_shape,
                  const T* offset_data, int offset, int deform_group,
                  int kernel_size, int stride, int pad, int dilation,
                  int zero_point, const VecInt& out_shape, T* out_data,
                  Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class DeformConvKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& offset,
                   const std::shared_ptr<Blob>& weight,
                   const std::shared_ptr<Blob>& bias,
                   std::shared_ptr<Blob>& output, Workspace* ws, int num_output,
                   int kernel_size, int stride, int pad, int dilation,
                   int group, int deform_group, bool bias_term,
                   int activate_type) = 0;
};

template <DeviceType D>
class DeformConvKernelDefault : public DeformConvKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& offset,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size, int stride, int pad,
           int dilation, int group, int deform_group, bool bias_term,
           int activate_type) override {
    int batch = input->shape(0), in_c = input->shape(1);

    int out_spatial_dim = output->count(2);
    int kernel_dim = kernel_size * kernel_size * in_c / group;

    int weight_offset = num_output * kernel_dim / group;
    int col_offset = kernel_dim * out_spatial_dim;
    int output_offset = num_output * out_spatial_dim / group;

    ws->GrowTempBuffer(kernel_dim * group * out_spatial_dim * sizeof(float));

    auto col_image = ws->CreateTempBlob({kernel_dim * group, out_spatial_dim},
                                        DataType::kF32);

    const auto* in_data = input->data<float>();
    const auto* offset_data = offset->data<float>();
    const auto* weight_data = weight->data<float>();
    const auto* bias_data = bias_term ? bias->data<float>() : nullptr;
    auto* out_data = output->mutable_data<float>();

    int in_num = input->num(), out_num = output->num(),
        out_count = output->count();

    for (int b = 0; b < batch; ++b) {
      Vision::DeformIm2Col<D, float>(
          in_data, input->shape(), offset_data, b * in_num, deform_group,
          kernel_size, stride, pad, dilation, 0, output->shape(),
          col_image->mutable_data<float>(), ws->Ctx());
      for (int g = 0; g < group; ++g) {
        Blas::BlasSgemm<D, float>(0, 0, num_output / group, out_spatial_dim,
                                  kernel_dim, 1, weight_data, weight_offset * g,
                                  col_image->data<float>(), col_offset * g, 0,
                                  out_data, b * out_num + output_offset * g,
                                  ws->Ctx());
      }
    }

    if (bias_term) {
      Vision::Bias<D, float>(out_data, out_count, bias_data, num_output,
                             out_spatial_dim, out_data, ws->Ctx());
    }

    if (activate_type == 1) {
      Vision::Activate<D, float>(out_data, out_data, out_count, activate_type,
                                 0, ws->Ctx());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_DEFORM_CONV_HPP_
