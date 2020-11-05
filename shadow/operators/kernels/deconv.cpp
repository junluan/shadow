#include "deconv.hpp"

namespace Shadow {

namespace Vision {

// check for 0 <= a < b
inline bool check_border(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <>
void Col2Im<DeviceType::kCPU, float>(const float* col_data,
                                     const VecInt& in_shape, int offset,
                                     int kernel_size_h, int kernel_size_w,
                                     int stride_h, int stride_w, int pad_h,
                                     int pad_w, int dilation,
                                     const VecInt& out_shape, float* in_data,
                                     Context* context) {
  in_data += offset;
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int spatial_dim = in_h * in_w;
  for (int k_c = 0; k_c < in_c; ++k_c, in_data += spatial_dim) {
    for (int k_s = 0; k_s < kernel_size_h * kernel_size_w; ++k_s) {
      int k_h = k_s / kernel_size_w;
      int k_w = k_s % kernel_size_w;
      int im_row = -pad_h + k_h * dilation;
      for (int h = 0; h < out_h; ++h, im_row += stride_h) {
        if (check_border(im_row, in_h)) {
          int im_col = -pad_w + k_w * dilation;
          for (int w = 0; w < out_w; ++w, ++col_data, im_col += stride_w) {
            if (check_border(im_col, in_w)) {
              in_data[im_row * in_w + im_col] += *(col_data);
            }
          }
        } else {
          col_data += out_w;
        }
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DeconvCPU, DeconvKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class DeconvKernelDNNL : public DeconvKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation,
           int group, bool bias_term, int activate_type) override {
    int in_c = input->shape(1);

    const auto& src_desc = idnnl::create_memory_desc<float>(
        input->shape(), dnnl::memory::format_tag::nchw);
    const auto& dst_desc = idnnl::create_memory_desc<float>(
        output->shape(), dnnl::memory::format_tag::nchw);
    dnnl::memory::desc weight_desc;
    if (group == 1) {
      weight_desc = idnnl::create_memory_desc<float>(
          {num_output, in_c, kernel_size_h, kernel_size_w},
          dnnl::memory::format_tag::iohw);
    } else {
      weight_desc = idnnl::create_memory_desc<float>(
          {group, num_output / group, in_c / group, kernel_size_h,
           kernel_size_w},
          dnnl::memory::format_tag::giohw);
    }
    const auto& bias_desc = idnnl::create_memory_desc<float>(
        {num_output}, bias_term ? dnnl::memory::format_tag::x
                                : dnnl::memory::format_tag::undef);

    const auto& deconv_desc = idnnl::create_deconvolution_desc(
        src_desc, weight_desc, bias_desc, dst_desc, pad_h, pad_w, stride_h,
        stride_w, dilation, dilation);

    idnnl::common_forward<dnnl::deconvolution_forward>(
        ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(), deconv_desc,
        input->data<float>(), weight->data<float>(),
        bias_term ? bias->data<float>() : nullptr,
        output->mutable_data<float>(), activate_type);
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(DeconvCPU, DeconvKernelDNNL);

#endif

}  // namespace Shadow
