#ifndef SHADOW_OPERATORS_KERNELS_DECONV_HPP_
#define SHADOW_OPERATORS_KERNELS_DECONV_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

#include "activate.hpp"
#include "scale.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Col2Im(const T* col_data, const VecInt& in_shape, int offset,
            int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
            int pad_h, int pad_w, int dilation, const VecInt& out_shape,
            T* in_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class DeconvKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& weight,
                   const std::shared_ptr<Blob>& bias,
                   std::shared_ptr<Blob>& output, Workspace* ws, int num_output,
                   int kernel_size_h, int kernel_size_w, int stride_h,
                   int stride_w, int pad_h, int pad_w, int dilation, int group,
                   bool bias_term, int activate_type) = 0;
};

template <DeviceType D>
class DeconvKernelDefault : public DeconvKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation,
           int group, bool bias_term, int activate_type) override {
    int batch = input->shape(0), in_c = input->shape(1);

    int conv_out_spatial_dim = input->count(2);
    int out_spatial_dim = output->count(2);
    int kernel_dim = kernel_size_h * kernel_size_w * num_output / group;

    int weight_offset = in_c * kernel_dim / group;
    int col_offset = kernel_dim * conv_out_spatial_dim;
    int output_offset = in_c * conv_out_spatial_dim / group;

    ws->GrowTempBuffer(kernel_dim * group * conv_out_spatial_dim *
                       sizeof(float));

    auto col_image = ws->CreateTempBlob(
        {kernel_dim * group, conv_out_spatial_dim}, DataType::kF32);

    const auto* in_data = input->data<float>();
    const auto* weight_data = weight->data<float>();
    const auto* bias_data = bias_term ? bias->data<float>() : nullptr;
    auto* out_data = output->mutable_data<float>();

    int in_num = input->num(), out_num = output->num(),
        out_count = output->count();

    Blas::Set<D, float>(out_count, 0, out_data, 0, ws->Ctx());

    for (int b = 0; b < batch; ++b) {
      for (int g = 0; g < group; ++g) {
        Blas::BlasSgemm<D, float>(
            1, 0, kernel_dim, conv_out_spatial_dim, in_c / group, 1,
            weight_data, weight_offset * g, in_data,
            b * in_num + output_offset * g, 0, col_image->mutable_data<float>(),
            col_offset * g, ws->Ctx());
      }
      Vision::Col2Im<D, float>(col_image->data<float>(), output->shape(),
                               b * out_num, kernel_size_h, kernel_size_w,
                               stride_h, stride_w, pad_h, pad_w, dilation,
                               input->shape(), out_data, ws->Ctx());
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

#endif  // SHADOW_OPERATORS_KERNELS_DECONV_HPP_
