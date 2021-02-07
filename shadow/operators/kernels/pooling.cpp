#include "pooling.hpp"

namespace Shadow {

namespace Vision {

template <>
void Pooling2D<DeviceType::kCPU, float>(const float* in_data,
                                        const VecInt& in_shape, int pool_type,
                                        int kernel_size_h, int kernel_size_w,
                                        int stride_h, int stride_w, int pad_h,
                                        int pad_w, const VecInt& out_shape,
                                        float* out_data, Context* context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c, in_data += in_h * in_w) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int hstart = h * stride_h - pad_h;
          int wstart = w * stride_w - pad_w;
          int hend = std::min(hstart + kernel_size_h, in_h + pad_h);
          int wend = std::min(wstart + kernel_size_w, in_w + pad_w);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0), wstart = std::max(wstart, 0);
          hend = std::min(hend, in_h), wend = std::min(wend, in_w);
          auto max_val = std::numeric_limits<float>::lowest(), sum_val = 0.f;
          for (int ph = hstart; ph < hend; ++ph) {
            for (int pw = wstart; pw < wend; ++pw) {
              auto value = in_data[ph * in_w + pw];
              max_val = std::max(max_val, value);
              sum_val += value;
            }
          }
          *out_data++ = (pool_type == 0) ? max_val : sum_val / pool_size;
        }
      }
    }
  }
}

template <>
void Pooling3D<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, int pool_type,
    int kernel_size_d, int kernel_size_h, int kernel_size_w, int stride_d,
    int stride_h, int stride_w, int pad_d, int pad_h, int pad_w,
    const VecInt& out_shape, float* out_data, Context* context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_d = in_shape[2], in_h = in_shape[3], in_w = in_shape[4];
  int out_d = out_shape[2], out_h = out_shape[3], out_w = out_shape[4];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c, in_data += in_d * in_h * in_w) {
      for (int d = 0; d < out_d; ++d) {
        for (int h = 0; h < out_h; ++h) {
          for (int w = 0; w < out_w; ++w) {
            int dstart = d * stride_d - pad_d;
            int hstart = h * stride_h - pad_h;
            int wstart = w * stride_w - pad_w;
            int dend = std::min(dstart + kernel_size_d, in_d + pad_d);
            int hend = std::min(hstart + kernel_size_h, in_h + pad_h);
            int wend = std::min(wstart + kernel_size_w, in_w + pad_w);
            int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            dstart = std::max(dstart, 0), hstart = std::max(hstart, 0),
            wstart = std::max(wstart, 0);
            dend = std::min(dend, in_d), hend = std::min(hend, in_h),
            wend = std::min(wend, in_w);
            auto max_val = std::numeric_limits<float>::lowest(), sum_val = 0.f;
            for (int pd = dstart; pd < dend; ++pd) {
              for (int ph = hstart; ph < hend; ++ph) {
                for (int pw = wstart; pw < wend; ++pw) {
                  auto value = in_data[(pd * in_h + ph) * in_w + pw];
                  max_val = std::max(max_val, value);
                  sum_val += value;
                }
              }
            }
            *out_data++ = (pool_type == 0) ? max_val : sum_val / pool_size;
          }
        }
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PoolingCPU, PoolingKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class PoolingKernelDNNL : public PoolingKernel {
 public:
  PoolingKernelDNNL() {
    default_kernel_ =
        std::make_shared<PoolingKernelDefault<DeviceType::kCPU>>();
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int pool_type, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w,
           bool full_pooling) override {
    if (full_pooling) {
      default_kernel_->Run(input, output, ws, pool_type, kernel_size_h,
                           kernel_size_w, stride_h, stride_w, pad_h, pad_w,
                           full_pooling);
    } else {
      const auto& src_desc = idnnl::create_memory_desc<float>(
          input->shape(), dnnl::memory::format_tag::nchw);
      const auto& dst_desc = idnnl::create_memory_desc<float>(
          output->shape(), dnnl::memory::format_tag::nchw);

      idnnl::common_forward<dnnl::pooling_forward>(
          ws->Ctx()->dnnl_handle(),
          dnnl::pooling_forward::desc(
              dnnl::prop_kind::forward_inference, get_algorithm(pool_type),
              src_desc, dst_desc, {stride_h, stride_w},
              {kernel_size_h, kernel_size_w}, {pad_h, pad_w}, {pad_h, pad_w}),
          input->data<float>(), output->mutable_data<float>());
    }
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int pool_type, int kernel_size_d, int kernel_size_h,
           int kernel_size_w, int stride_d, int stride_h, int stride_w,
           int pad_d, int pad_h, int pad_w, bool full_pooling) override {
    if (full_pooling) {
      default_kernel_->Run(input, output, ws, pool_type, kernel_size_d,
                           kernel_size_h, kernel_size_w, stride_d, stride_h,
                           stride_w, pad_d, pad_h, pad_w, full_pooling);
    } else {
      const auto& src_desc = idnnl::create_memory_desc<float>(
          input->shape(), dnnl::memory::format_tag::ncdhw);
      const auto& dst_desc = idnnl::create_memory_desc<float>(
          output->shape(), dnnl::memory::format_tag::ncdhw);

      idnnl::common_forward<dnnl::pooling_forward>(
          ws->Ctx()->dnnl_handle(),
          dnnl::pooling_forward::desc(
              dnnl::prop_kind::forward_inference, get_algorithm(pool_type),
              src_desc, dst_desc, {stride_d, stride_h, stride_w},
              {kernel_size_d, kernel_size_h, kernel_size_w},
              {pad_d, pad_h, pad_w}, {pad_d, pad_h, pad_w}),
          input->data<float>(), output->mutable_data<float>());
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }

 private:
  static dnnl::algorithm get_algorithm(int pool_type) {
    switch (pool_type) {
      case 0:
        return dnnl::algorithm::pooling_max;
      case 1:
        return dnnl::algorithm::pooling_avg_include_padding;
      default:
        return dnnl::algorithm::undef;
    }
  }

  std::shared_ptr<PoolingKernelDefault<DeviceType::kCPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_DNNL(PoolingCPU, PoolingKernelDNNL);

#endif

}  // namespace Shadow
