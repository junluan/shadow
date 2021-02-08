#include "conv.hpp"

namespace Shadow {

namespace Vision {

// check for 0 <= a < b
inline bool check_border(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <>
void Im2Col2D<DeviceType::kCPU, float>(const float* in_data,
                                       const VecInt& in_shape,
                                       int kernel_size_h, int kernel_size_w,
                                       int stride_h, int stride_w, int pad_h,
                                       int pad_w, int dilation_h,
                                       int dilation_w, const VecInt& out_shape,
                                       float* col_data, Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int c = 0; c < in_c; ++c, in_data += in_h * in_w) {
    for (int k_s = 0; k_s < kernel_size_h * kernel_size_w; ++k_s) {
      int kh = k_s / kernel_size_w, kw = k_s % kernel_size_w;
      int h_in = kh * dilation_h - pad_h;
      for (int h = 0; h < out_h; ++h, h_in += stride_h) {
        if (check_border(h_in, in_h)) {
          int w_in = kw * dilation_w - pad_w;
          for (int w = 0; w < out_w; ++w, w_in += stride_w) {
            if (check_border(w_in, in_w)) {
              *col_data++ = in_data[h_in * in_w + w_in];
            } else {
              *col_data++ = 0.f;
            }
          }
        } else {
          for (int w = 0; w < out_w; ++w) {
            *col_data++ = 0.f;
          }
        }
      }
    }
  }
}

template <>
void Im2Col3D<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, int kernel_size_d,
    int kernel_size_h, int kernel_size_w, int stride_d, int stride_h,
    int stride_w, int pad_d, int pad_h, int pad_w, int dilation_d,
    int dilation_h, int dilation_w, const VecInt& out_shape, float* col_data,
    Context* context) {
  int in_c = in_shape[1], in_d = in_shape[2], in_h = in_shape[3],
      in_w = in_shape[4];
  int out_d = out_shape[2], out_h = out_shape[3], out_w = out_shape[4];
  for (int c = 0; c < in_c; ++c, in_data += in_d * in_h * in_w) {
    for (int k_s = 0; k_s < kernel_size_d * kernel_size_h * kernel_size_w;
         ++k_s) {
      int temp = k_s / kernel_size_w;
      int kd = temp / kernel_size_h, kh = temp % kernel_size_h,
          kw = k_s % kernel_size_w;
      int d_in = kd * dilation_d - pad_d;
      for (int d = 0; d < out_d; ++d, d_in += stride_d) {
        if (check_border(d_in, in_d)) {
          int h_in = kh * dilation_h - pad_h;
          for (int h = 0; h < out_h; ++h, h_in += stride_h) {
            if (check_border(h_in, in_h)) {
              int w_in = kw * dilation_w - pad_w;
              for (int w = 0; w < out_w; ++w, w_in += stride_w) {
                if (check_border(w_in, in_w)) {
                  *col_data++ = in_data[(d_in * in_h + h_in) * in_w + w_in];
                } else {
                  *col_data++ = 0.f;
                }
              }
            } else {
              for (int w = 0; w < out_w; ++w) {
                *col_data++ = 0.f;
              }
            }
          }
        } else {
          for (int n = 0; n < out_h * out_w; ++n) {
            *col_data++ = 0.f;
          }
        }
      }
    }
  }
}

template <>
void Depthwise2D<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, const float* weight_data,
    const float* bias_data, int kernel_size_h, int kernel_size_w, int stride_h,
    int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w,
    bool bias_term, const VecInt& out_shape, float* out_data,
    Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_c = out_shape[1], out_h = out_shape[2], out_w = out_shape[3];
  int k = out_c / in_c;
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < out_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          const auto* weight_data_ptr =
              weight_data + c * kernel_size_h * kernel_size_w;
          double sum_val = 0;
          for (int kh = 0; kh < kernel_size_h; ++kh) {
            int h_in = h * stride_h - pad_h + kh * dilation_h;
            for (int kw = 0; kw < kernel_size_w; ++kw) {
              int w_in = w * stride_w - pad_w + kw * dilation_w;
              if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                sum_val += in_data[h_in * in_w + w_in] * *weight_data_ptr;
              }
              weight_data_ptr++;
            }
          }
          if (bias_term) {
            sum_val += bias_data[c];
          }
          *out_data++ = static_cast<float>(sum_val);
        }
      }
      if ((c + 1) % k == 0) {
        in_data += in_h * in_w;
      }
    }
  }
}

template <>
void Depthwise3D<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, const float* weight_data,
    const float* bias_data, int kernel_size_d, int kernel_size_h,
    int kernel_size_w, int stride_d, int stride_h, int stride_w, int pad_d,
    int pad_h, int pad_w, int dilation_d, int dilation_h, int dilation_w,
    bool bias_term, const VecInt& out_shape, float* out_data,
    Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_d = in_shape[2], in_h = in_shape[3],
      in_w = in_shape[4];
  int out_c = out_shape[1], out_d = out_shape[2], out_h = out_shape[3],
      out_w = out_shape[4];
  int k = out_c / in_c;
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < out_c; ++c) {
      for (int d = 0; d < out_d; ++d) {
        for (int h = 0; h < out_h; ++h) {
          for (int w = 0; w < out_w; ++w) {
            const auto* weight_data_ptr =
                weight_data + c * kernel_size_d * kernel_size_h * kernel_size_w;
            double sum_val = 0;
            for (int kd = 0; kd < kernel_size_d; ++kd) {
              int d_in = d * stride_d - pad_d + kd * dilation_d;
              for (int kh = 0; kh < kernel_size_h; ++kh) {
                int h_in = h * stride_h - pad_h + kh * dilation_h;
                for (int kw = 0; kw < kernel_size_w; ++kw) {
                  int w_in = w * stride_w - pad_w + kw * dilation_w;
                  if (d_in >= 0 && d_in < in_d && h_in >= 0 && h_in < in_h &&
                      w_in >= 0 && w_in < in_w) {
                    sum_val += in_data[(d_in * in_h + h_in) * in_w + w_in] *
                               *weight_data_ptr;
                  }
                  weight_data_ptr++;
                }
              }
            }
            if (bias_term) {
              sum_val += bias_data[c];
            }
            *out_data++ = static_cast<float>(sum_val);
          }
        }
      }
      if ((c + 1) % k == 0) {
        in_data += in_d * in_h * in_w;
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
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
           int dilation_w, int group, bool bias_term,
           int activate_type) override {
    int batch = input->shape(0), in_c = input->shape(1), in_h = input->shape(2),
        in_w = input->shape(3);

    if (batch == 1 && group == 1 && dilation_h == 1 && dilation_w == 1 &&
        bias_term) {
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
                           pad_h, pad_w, dilation_h, dilation_w, group,
                           bias_term, activate_type);
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
    default_kernel_->Run(input, weight, bias, output, ws, num_output,
                         kernel_size_d, kernel_size_h, kernel_size_w, stride_d,
                         stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d,
                         dilation_h, dilation_w, group, bias_term,
                         activate_type);
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

    const auto& conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
        src_desc, weight_desc, bias_desc, dst_desc, {stride_h, stride_w},
        {dilation_h - 1, dilation_w - 1}, {pad_h, pad_w}, {pad_h, pad_w});

    idnnl::common_forward<dnnl::convolution_forward>(
        ws->Ctx()->dnnl_handle(), conv_desc, input->data<float>(),
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
          dnnl::memory::format_tag::oidhw);
    } else {
      weight_desc = idnnl::create_memory_desc<float>(
          {group, num_output / group, in_c / group, kernel_size_d,
           kernel_size_h, kernel_size_w},
          dnnl::memory::format_tag::goidhw);
    }
    const auto& bias_desc = idnnl::create_memory_desc<float>(
        {num_output}, bias_term ? dnnl::memory::format_tag::x
                                : dnnl::memory::format_tag::undef);

    const auto& conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto,
        src_desc, weight_desc, bias_desc, dst_desc,
        {stride_d, stride_h, stride_w},
        {dilation_d - 1, dilation_h - 1, dilation_w - 1}, {pad_d, pad_h, pad_w},
        {pad_d, pad_h, pad_w});

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
