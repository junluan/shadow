#include "deconv.hpp"

namespace Shadow {

namespace Vision {

template <>
void Col2Im2D<DeviceType::kCPU, float>(const float* col_data,
                                       const VecInt& in_shape,
                                       int kernel_size_h, int kernel_size_w,
                                       int stride_h, int stride_w, int pad_h,
                                       int pad_w, int dilation_h,
                                       int dilation_w, const VecInt& out_shape,
                                       float* out_data, Context* context) {
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_c = out_shape[1], out_h = out_shape[2], out_w = out_shape[3];
  int ke_h = (kernel_size_h - 1) * dilation_h + 1;
  int ke_w = (kernel_size_w - 1) * dilation_w + 1;
  for (int c = 0; c < out_c;
       ++c, col_data += in_h * in_w * kernel_size_h * kernel_size_w) {
    for (int h = pad_h; h < out_h + pad_h; ++h) {
      for (int w = pad_w; w < out_w + pad_w; ++w) {
        int hstart = h < ke_h ? 0 : (h - ke_h) / stride_h + 1;
        int hend = std::min(h / stride_h + 1, in_h);
        int wstart = w < ke_w ? 0 : (w - ke_w) / stride_w + 1;
        int wend = std::min(w / stride_w + 1, in_w);
        double sum_val = 0;
        for (int h_in = hstart; h_in < hend; ++h_in) {
          int h_k = h - h_in * stride_h;
          if (h_k % dilation_h == 0) {
            for (int w_in = wstart; w_in < wend; ++w_in) {
              int w_k = w - w_in * stride_w;
              if (w_k % dilation_w == 0) {
                sum_val += col_data[((h_k / dilation_h * kernel_size_w +
                                      w_k / dilation_w) *
                                         in_h +
                                     h_in) *
                                        in_w +
                                    w_in];
              }
            }
          }
        }
        *out_data++ = static_cast<float>(sum_val);
      }
    }
  }
}

template <>
void Col2Im3D<DeviceType::kCPU, float>(
    const float* col_data, const VecInt& in_shape, int kernel_size_d,
    int kernel_size_h, int kernel_size_w, int stride_d, int stride_h,
    int stride_w, int pad_d, int pad_h, int pad_w, int dilation_d,
    int dilation_h, int dilation_w, const VecInt& out_shape, float* out_data,
    Context* context) {
  int in_d = in_shape[2], in_h = in_shape[3], in_w = in_shape[4];
  int out_c = out_shape[1], out_d = out_shape[2], out_h = out_shape[3],
      out_w = out_shape[4];
  int ke_d = (kernel_size_d - 1) * dilation_d + 1;
  int ke_h = (kernel_size_h - 1) * dilation_h + 1;
  int ke_w = (kernel_size_w - 1) * dilation_w + 1;
  for (int c = 0; c < out_c; ++c, col_data += in_d * in_h * in_w *
                                              kernel_size_d * kernel_size_h *
                                              kernel_size_w) {
    for (int d = pad_d; d < out_d + pad_d; ++d) {
      for (int h = pad_h; h < out_h + pad_h; ++h) {
        for (int w = pad_w; w < out_w + pad_w; ++w) {
          int dstart = d < ke_d ? 0 : (d - ke_d) / stride_d + 1;
          int dend = std::min(d / stride_d + 1, in_d);
          int hstart = h < ke_h ? 0 : (h - ke_h) / stride_h + 1;
          int hend = std::min(h / stride_h + 1, in_h);
          int wstart = w < ke_w ? 0 : (w - ke_w) / stride_w + 1;
          int wend = std::min(w / stride_w + 1, in_w);
          double sum_val = 0;
          for (int d_in = dstart; d_in < dend; ++d_in) {
            int d_k = d - d_in * stride_d;
            if (d_k % dilation_d == 0) {
              for (int h_in = hstart; h_in < hend; ++h_in) {
                int h_k = h - h_in * stride_h;
                if (h_k % dilation_h == 0) {
                  for (int w_in = wstart; w_in < wend; ++w_in) {
                    int w_k = w - w_in * stride_w;
                    if (w_k % dilation_w == 0) {
                      sum_val += col_data[((((d_k / dilation_d * kernel_size_h +
                                              h_k / dilation_h) *
                                                 kernel_size_w +
                                             w_k / dilation_w) *
                                                in_d +
                                            d_in) *
                                               in_h +
                                           h_in) *
                                              in_w +
                                          w_in];
                    }
                  }
                }
              }
            }
          }
          *out_data++ = static_cast<float>(sum_val);
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
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
           int dilation_w, int group, bool bias_term,
           int activate_type) override {
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

    const auto& deconv_desc = dnnl::deconvolution_forward::desc(
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::deconvolution_direct, src_desc, weight_desc, bias_desc,
        dst_desc, {stride_h, stride_w}, {dilation_h - 1, dilation_w - 1},
        {pad_h, pad_w}, {pad_h, pad_w});

    idnnl::common_forward<dnnl::deconvolution_forward>(
        ws->Ctx()->dnnl_handle(), deconv_desc, input->data<float>(),
        weight->data<float>(), bias_term ? bias->data<float>() : nullptr,
        output->mutable_data<float>(), activate_type);
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_d, int kernel_size_h,
           int kernel_size_w, int stride_d, int stride_h, int stride_w,
           int pad_d, int pad_h, int pad_w, int dilation_d, int dilation_h,
           int dilation_w, int group, bool bias_term,
           int activate_type) override {
    int in_c = input->shape(1);

    const auto& src_desc = idnnl::create_memory_desc<float>(
        input->shape(), dnnl::memory::format_tag::ncdhw);
    const auto& dst_desc = idnnl::create_memory_desc<float>(
        output->shape(), dnnl::memory::format_tag::ncdhw);
    dnnl::memory::desc weight_desc;
    if (group == 1) {
      weight_desc = idnnl::create_memory_desc<float>(
          {num_output, in_c, kernel_size_d, kernel_size_h, kernel_size_w},
          dnnl::memory::format_tag::iodhw);
    } else {
      weight_desc = idnnl::create_memory_desc<float>(
          {group, num_output / group, in_c / group, kernel_size_d,
           kernel_size_h, kernel_size_w},
          dnnl::memory::format_tag::giodhw);
    }
    const auto& bias_desc = idnnl::create_memory_desc<float>(
        {num_output}, bias_term ? dnnl::memory::format_tag::x
                                : dnnl::memory::format_tag::undef);

    const auto& deconv_desc = dnnl::deconvolution_forward::desc(
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::deconvolution_direct, src_desc, weight_desc, bias_desc,
        dst_desc, {stride_d, stride_h, stride_w},
        {dilation_d - 1, dilation_h - 1, dilation_w - 1}, {pad_d, pad_h, pad_w},
        {pad_d, pad_h, pad_w});

    idnnl::common_forward<dnnl::deconvolution_forward>(
        ws->Ctx()->dnnl_handle(), deconv_desc, input->data<float>(),
        weight->data<float>(), bias_term ? bias->data<float>() : nullptr,
        output->mutable_data<float>(), activate_type);
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(DeconvCPU, DeconvKernelDNNL);

#endif

}  // namespace Shadow
