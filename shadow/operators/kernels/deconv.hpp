#ifndef SHADOW_OPERATORS_KERNELS_DECONV_HPP_
#define SHADOW_OPERATORS_KERNELS_DECONV_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

#include "activate.hpp"
#include "scale.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Col2Im2D(const T* col_data, const VecInt& in_shape, int kernel_size_h,
              int kernel_size_w, int stride_h, int stride_w, int pad_h,
              int pad_w, int dilation_h, int dilation_w,
              const VecInt& out_shape, T* out_data, Context* context);

template <DeviceType D, typename T>
void Col2Im3D(const T* col_data, const VecInt& in_shape, int kernel_size_d,
              int kernel_size_h, int kernel_size_w, int stride_d, int stride_h,
              int stride_w, int pad_d, int pad_h, int pad_w, int dilation_d,
              int dilation_h, int dilation_w, const VecInt& out_shape,
              T* out_data, Context* context);

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
                   int stride_w, int pad_h, int pad_w, int dilation_h,
                   int dilation_w, int group, bool bias_term,
                   int activate_type) = 0;

  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& weight,
                   const std::shared_ptr<Blob>& bias,
                   std::shared_ptr<Blob>& output, Workspace* ws, int num_output,
                   int kernel_size_d, int kernel_size_h, int kernel_size_w,
                   int stride_d, int stride_h, int stride_w, int pad_d,
                   int pad_h, int pad_w, int dilation_d, int dilation_h,
                   int dilation_w, int group, bool bias_term,
                   int activate_type){};
};

template <DeviceType D>
class DeconvKernelDefault : public DeconvKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
           int dilation_w, int group, bool bias_term,
           int activate_type) override {
    const auto* in_data = input->data<float>();
    const auto* weight_data = weight->data<float>();
    const auto* bias_data = bias_term ? bias->data<float>() : nullptr;
    auto* out_data = output->mutable_data<float>();

    int in_c = input->shape(1);

    int in_spatial_dim = input->count(2);
    int kernel_dim = kernel_size_h * kernel_size_w * num_output / group;

    int weight_offset = in_c * kernel_dim / group;
    int col_offset = kernel_dim * in_spatial_dim;
    int input_offset = in_c * in_spatial_dim / group;

    ws->GrowTempBuffer(kernel_dim * group * in_spatial_dim * sizeof(float));

    auto col_image = ws->CreateTempBlob({kernel_dim * group, in_spatial_dim},
                                        DataType::kF32);

    int batch = input->shape(0), in_num = input->num(), out_num = output->num();

    for (int b = 0; b < batch; ++b) {
      for (int g = 0; g < group; ++g) {
        Blas::BlasSgemm<D, float>(
            1, 0, kernel_dim, in_spatial_dim, in_c / group, 1, weight_data,
            weight_offset * g, in_data, b * in_num + input_offset * g, 0,
            col_image->mutable_data<float>(), col_offset * g, ws->Ctx().get());
      }
      Vision::Col2Im2D<D, float>(
          col_image->data<float>(), input->shape(), kernel_size_h,
          kernel_size_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
          dilation_w, output->shape(), out_data + b * out_num, ws->Ctx().get());
    }

    if (bias_term) {
      Vision::Bias<D, float>(out_data, output->count(), bias_data, num_output,
                             output->count(2), out_data, ws->Ctx().get());
    }

    if (activate_type == 1) {
      Vision::Activate<D, float>(out_data, out_data, output->count(),
                                 activate_type, 0, ws->Ctx().get());
    }
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_d, int kernel_size_h,
           int kernel_size_w, int stride_d, int stride_h, int stride_w,
           int pad_d, int pad_h, int pad_w, int dilation_d, int dilation_h,
           int dilation_w, int group, bool bias_term,
           int activate_type) override {
    const auto* in_data = input->data<float>();
    const auto* weight_data = weight->data<float>();
    const auto* bias_data = bias_term ? bias->data<float>() : nullptr;
    auto* out_data = output->mutable_data<float>();

    int in_c = input->shape(1);

    int in_spatial_dim = input->count(2);
    int kernel_dim =
        kernel_size_d * kernel_size_h * kernel_size_w * num_output / group;

    int weight_offset = in_c * kernel_dim / group;
    int col_offset = kernel_dim * in_spatial_dim;
    int input_offset = in_c * in_spatial_dim / group;

    ws->GrowTempBuffer(kernel_dim * group * in_spatial_dim * sizeof(float));

    auto col_image = ws->CreateTempBlob({kernel_dim * group, in_spatial_dim},
                                        DataType::kF32);

    int batch = input->shape(0), in_num = input->num(), out_num = output->num();

    for (int b = 0; b < batch; ++b) {
      for (int g = 0; g < group; ++g) {
        Blas::BlasSgemm<D, float>(
            1, 0, kernel_dim, in_spatial_dim, in_c / group, 1, weight_data,
            weight_offset * g, in_data, b * in_num + input_offset * g, 0,
            col_image->mutable_data<float>(), col_offset * g, ws->Ctx().get());
      }
      Vision::Col2Im3D<D, float>(
          col_image->data<float>(), input->shape(), kernel_size_d,
          kernel_size_h, kernel_size_w, stride_d, stride_h, stride_w, pad_d,
          pad_h, pad_w, dilation_d, dilation_h, dilation_w, output->shape(),
          out_data + b * out_num, ws->Ctx().get());
    }

    if (bias_term) {
      Vision::Bias<D, float>(out_data, output->count(), bias_data, num_output,
                             output->count(2), out_data, ws->Ctx().get());
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

#endif  // SHADOW_OPERATORS_KERNELS_DECONV_HPP_
