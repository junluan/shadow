#ifndef SHADOW_OPERATORS_KERNELS_DEFORM_CONV_HPP_
#define SHADOW_OPERATORS_KERNELS_DEFORM_CONV_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

#include "activate.hpp"
#include "scale.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void DeformIm2Col2D(const T* in_data, const VecInt& in_shape,
                    const T* offset_data, const T* mask_data, int kernel_size_h,
                    int kernel_size_w, int stride_h, int stride_w, int pad_h,
                    int pad_w, int dilation_h, int dilation_w, int deform_group,
                    bool use_mask, const VecInt& out_shape, T* col_data,
                    Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class DeformConvKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& offset,
                   const std::shared_ptr<Blob>& mask,
                   const std::shared_ptr<Blob>& weight,
                   const std::shared_ptr<Blob>& bias,
                   std::shared_ptr<Blob>& output, Workspace* ws, int num_output,
                   int kernel_size_h, int kernel_size_w, int stride_h,
                   int stride_w, int pad_h, int pad_w, int dilation_h,
                   int dilation_w, int group, int deform_group, bool use_mask,
                   bool bias_term, int activate_type) = 0;
};

template <DeviceType D>
class DeformConvKernelDefault : public DeformConvKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& offset,
           const std::shared_ptr<Blob>& mask,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
           int dilation_w, int group, int deform_group, bool use_mask,
           bool bias_term, int activate_type) override {
    const auto* in_data = input->data<float>();
    const auto* offset_data = offset->data<float>();
    const auto* mask_data = use_mask ? mask->data<float>() : nullptr;
    const auto* weight_data = weight->data<float>();
    const auto* bias_data = bias_term ? bias->data<float>() : nullptr;
    auto* out_data = output->mutable_data<float>();

    int in_c = input->shape(1);

    int out_spatial_dim = output->count(2);
    int kernel_dim = kernel_size_h * kernel_size_w * in_c / group;

    int weight_offset = num_output * kernel_dim / group;
    int col_offset = kernel_dim * out_spatial_dim;
    int output_offset = num_output * out_spatial_dim / group;

    ws->GrowTempBuffer(kernel_dim * group * out_spatial_dim * sizeof(float));

    auto col_image = ws->CreateTempBlob({kernel_dim * group, out_spatial_dim},
                                        DataType::kF32);

    int batch = input->shape(0), in_num = input->num(),
        offset_num = offset->num(), mask_num = use_mask ? mask->num() : 0,
        out_num = output->num();

    for (int b = 0; b < batch; ++b) {
      Vision::DeformIm2Col2D<D, float>(
          in_data + b * in_num, input->shape(), offset_data + b * offset_num,
          use_mask ? mask_data + b * mask_num : nullptr, kernel_size_h,
          kernel_size_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
          dilation_w, deform_group, use_mask, output->shape(),
          col_image->mutable_data<float>(), ws->Ctx().get());
      for (int g = 0; g < group; ++g) {
        Blas::BlasSgemm<D, float>(0, 0, num_output / group, out_spatial_dim,
                                  kernel_dim, 1, weight_data, weight_offset * g,
                                  col_image->data<float>(), col_offset * g, 0,
                                  out_data, b * out_num + output_offset * g,
                                  ws->Ctx().get());
      }
    }

    if (bias_term) {
      Vision::Bias<D, float>(out_data, output->count(), bias_data, num_output,
                             out_spatial_dim, out_data, ws->Ctx().get());
    }

    if (activate_type == 1) {
      Vision::Activate<D, float>(out_data, out_data, output->count(),
                                 activate_type, 0, ws->Ctx().get());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_DEFORM_CONV_HPP_
