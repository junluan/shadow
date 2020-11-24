#include "conv.hpp"

namespace Shadow {

namespace Vision {

// check for 0 <= a < b
inline bool check_border(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <>
void Im2Col<DeviceType::kCPU, float>(const float* in_data,
                                     const VecInt& in_shape, int offset,
                                     int kernel_size_h, int kernel_size_w,
                                     int stride_h, int stride_w, int pad_h,
                                     int pad_w, int dilation, int zero_point,
                                     const VecInt& out_shape, float* col_data,
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
          for (int w = 0; w < out_w; ++w, im_col += stride_w) {
            if (check_border(im_col, in_w)) {
              *(col_data++) = in_data[im_row * in_w + im_col];
            } else {
              *(col_data++) = static_cast<float>(zero_point);
            }
          }
        } else {
          for (int w = 0; w < out_w; ++w) {
            *(col_data++) = static_cast<float>(zero_point);
          }
        }
      }
    }
  }
}

template <>
void Depthwise<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, const float* weight_data,
    const float* bias_data, int kernel_size_h, int kernel_size_w, int stride_h,
    int stride_w, int pad_h, int pad_w, int dilation, bool bias_term,
    const VecInt& out_shape, float* out_data, Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      const auto* in_offset_data = in_data + (b * in_c + c) * in_h * in_w;
      auto* out_offset_data = out_data + (b * in_c + c) * out_h * out_w;
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          const auto* weight_offset_data =
              weight_data + c * kernel_size_h * kernel_size_w;
          double sum_val = 0;
          for (int kh = 0; kh < kernel_size_h; ++kh) {
            for (int kw = 0; kw < kernel_size_w; ++kw) {
              int h_in = h * stride_h - pad_h + kh * dilation;
              int w_in = w * stride_w - pad_w + kw * dilation;
              if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                sum_val +=
                    in_offset_data[h_in * in_w + w_in] * *weight_offset_data;
              }
              weight_offset_data++;
            }
          }
          if (bias_term) {
            sum_val += bias_data[c];
          }
          out_offset_data[h * out_w + w] = static_cast<float>(sum_val);
        }
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ConvCPU, ConvKernelDefault<DeviceType::kCPU>);

#if defined(USE_NNPACK)

class ConvKernelNNPACK : public ConvKernel {
 public:
  ConvKernelNNPACK() {
    default_kernel_ = std::make_shared<ConvKernelDefault<DeviceType::kCPU>>();
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation,
           int group, bool bias_term, int activate_type) override {
    int batch = input->shape(0), in_c = input->shape(1), in_h = input->shape(2),
        in_w = input->shape(3);

    if (batch == 1 && group == 1 && dilation == 1 && bias_term) {
      auto nnp_activation =
          activate_type == 1 ? nnp_activation_relu : nnp_activation_identity;
      auto nnp_input_size =
          nnp_size{static_cast<size_t>(in_w), static_cast<size_t>(in_h)};
      auto nnp_kernel_size = nnp_size{static_cast<size_t>(kernel_size_w),
                                      static_cast<size_t>(kernel_size_h)};
      auto nnp_stride = nnp_size{static_cast<size_t>(stride_w),
                                 static_cast<size_t>(stride_h)};
      auto nnp_pad = nnp_padding{};
      nnp_pad.top = nnp_pad.bottom = static_cast<size_t>(pad_h);
      nnp_pad.left = nnp_pad.right = static_cast<size_t>(pad_w);

      int out_c = output->shape(1);
      auto status = nnp_convolution_inference(
          nnp_convolution_algorithm_auto,
          nnp_convolution_transform_strategy_compute, in_c, out_c,
          nnp_input_size, nnp_pad, nnp_kernel_size, nnp_stride,
          input->data<float>(), weight->data<float>(), bias->data<float>(),
          output->mutable_data<float>(), nullptr, nullptr, nnp_activation,
          nullptr, pthreadpool_t(ws->Ctx()->nnpack_handle()), nullptr);
      CHECK_EQ(nnp_status_success, status);
    } else {
      default_kernel_->Run(input, weight, bias, output, ws, num_output,
                           kernel_size_h, kernel_size_w, stride_h, stride_w,
                           pad_h, pad_w, dilation, group, bias_term,
                           activate_type);
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "NNPACK"; }

 private:
  std::shared_ptr<ConvKernelDefault<DeviceType::kCPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL(ConvCPU(NNPACK), ConvKernelNNPACK);

#endif

#if defined(USE_DNNL)

class ConvKernelDNNL : public ConvKernel {
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
          dnnl::memory::format_tag::oihw);
    } else {
      weight_desc = idnnl::create_memory_desc<float>(
          {group, num_output / group, in_c / group, kernel_size_h,
           kernel_size_w},
          dnnl::memory::format_tag::goihw);
    }
    const auto& bias_desc = idnnl::create_memory_desc<float>(
        {num_output}, bias_term ? dnnl::memory::format_tag::x
                                : dnnl::memory::format_tag::undef);

    const auto& conv_desc = idnnl::create_convolution_desc(
        src_desc, weight_desc, bias_desc, dst_desc, pad_h, pad_w, stride_h,
        stride_w, dilation, dilation);

    idnnl::common_forward<dnnl::convolution_forward>(
        ws->Ctx()->dnnl_handle(), conv_desc, input->data<float>(),
        weight->data<float>(), bias_term ? bias->data<float>() : nullptr,
        output->mutable_data<float>(), activate_type);
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(ConvCPU, ConvKernelDNNL);

#endif

}  // namespace Shadow
