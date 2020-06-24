#include "pooling.hpp"

namespace Shadow {

namespace Vision {

template <>
void Pooling<DeviceType::kCPU, float>(const float* in_data,
                                      const VecInt& in_shape, int pool_type,
                                      int kernel_size_h, int kernel_size_w,
                                      int stride_h, int stride_w, int pad_h,
                                      int pad_w, const VecInt& out_shape,
                                      float* out_data, Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int kistart = h * stride_h - pad_h, kjstart = w * stride_w - pad_w;
          int kiend = std::min(kistart + kernel_size_h, in_h + pad_h);
          int kjend = std::min(kjstart + kernel_size_w, in_w + pad_w);
          int pool_size = (kiend - kistart) * (kjend - kjstart);
          kistart = std::max(kistart, 0), kjstart = std::max(kjstart, 0);
          kiend = std::min(kiend, in_h), kjend = std::min(kjend, in_w);
          auto max = std::numeric_limits<float>::lowest(), sum = 0.f;
          for (int ki = kistart; ki < kiend; ++ki) {
            for (int kj = kjstart; kj < kjend; ++kj) {
              int index = kj + in_w * (ki + in_h * (c + in_c * b));
              auto value = in_data[index];
              max = (value > max) ? value : max;
              sum += value;
            }
          }
          int out_index = w + out_w * (h + out_h * (c + in_c * b));
          out_data[out_index] = (pool_type == 0) ? max : sum / pool_size;
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
      const auto& src_desc = idnnl::create_memory_desc<float>(input->shape());
      const auto& dst_desc = idnnl::create_memory_desc<float>(output->shape());

      const auto& pooling_desc = idnnl::create_pooling_desc(
          src_desc, dst_desc, pool_type, kernel_size_h, kernel_size_w, stride_h,
          stride_w, pad_h, pad_w);

      idnnl::common_forward<dnnl::pooling_forward>(
          ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(), pooling_desc,
          input->data<float>(), output->mutable_data<float>());
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }

 private:
  std::shared_ptr<PoolingKernelDefault<DeviceType::kCPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_DNNL(PoolingCPU, PoolingKernelDNNL);

#endif

}  // namespace Shadow
