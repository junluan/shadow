#include "pooling.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelPooling2D(const float* in_data, int count, int channel,
                                int in_h, int in_w, int pool_type,
                                int kernel_size_h, int kernel_size_w,
                                int stride_h, int stride_w, int pad_h,
                                int pad_w, int out_h, int out_w,
                                float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    int hstart = h_out * stride_h - pad_h;
    int wstart = w_out * stride_w - pad_w;
    int hend = min(hstart + kernel_size_h, in_h + pad_h);
    int wend = min(wstart + kernel_size_w, in_w + pad_w);

    int pool_size = (hend - hstart) * (wend - wstart);

    hstart = max(hstart, 0), wstart = max(wstart, 0);
    hend = min(hend, in_h), wend = min(wend, in_w);

    in_data += (b_out * channel + c_out) * in_h * in_w;

    auto max_val = -FLT_MAX, sum_val = 0.f;
    for (int ph = hstart; ph < hend; ++ph) {
      for (int pw = wstart; pw < wend; ++pw) {
        auto value = in_data[ph * in_w + pw];
        max_val = fmaxf(max_val, value);
        sum_val += value;
      }
    }

    out_data[globalid] = (pool_type == 0) ? max_val : sum_val / pool_size;
  }
}

__global__ void KernelPooling3D(const float* in_data, int count, int channel,
                                int in_d, int in_h, int in_w, int pool_type,
                                int kernel_size_d, int kernel_size_h,
                                int kernel_size_w, int stride_d, int stride_h,
                                int stride_w, int pad_d, int pad_h, int pad_w,
                                int out_d, int out_h, int out_w,
                                float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp /= out_h;
    int d_out = temp % out_d;
    temp /= out_d;
    int c_out = temp % channel;
    int b_out = temp / channel;

    int dstart = d_out * stride_d - pad_d;
    int hstart = h_out * stride_h - pad_h;
    int wstart = w_out * stride_w - pad_w;
    int dend = min(dstart + kernel_size_d, in_d + pad_d);
    int hend = min(hstart + kernel_size_h, in_h + pad_h);
    int wend = min(wstart + kernel_size_w, in_w + pad_w);

    int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);

    dstart = max(dstart, 0), hstart = max(hstart, 0), wstart = max(wstart, 0);
    dend = min(dend, in_d), hend = min(hend, in_h), wend = min(wend, in_w);

    in_data += (b_out * channel + c_out) * in_d * in_h * in_w;

    auto max_val = -FLT_MAX, sum_val = 0.f;
    for (int pd = dstart; pd < dend; ++pd) {
      for (int ph = hstart; ph < hend; ++ph) {
        for (int pw = wstart; pw < wend; ++pw) {
          auto value = in_data[(pd * in_h + ph) * in_w + pw];
          max_val = fmaxf(max_val, value);
          sum_val += value;
        }
      }
    }

    out_data[globalid] = (pool_type == 0) ? max_val : sum_val / pool_size;
  }
}

template <>
void Pooling2D<DeviceType::kGPU, float>(const float* in_data,
                                        const VecInt& in_shape, int pool_type,
                                        int kernel_size_h, int kernel_size_w,
                                        int stride_h, int stride_w, int pad_h,
                                        int pad_w, const VecInt& out_shape,
                                        float* out_data, Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelPooling2D<<<GetBlocks(count), NumThreads, 0,
                    cudaStream_t(context->stream())>>>(
      in_data, count, in_c, in_h, in_w, pool_type, kernel_size_h, kernel_size_w,
      stride_h, stride_w, pad_h, pad_w, out_h, out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <>
void Pooling3D<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, int pool_type,
    int kernel_size_d, int kernel_size_h, int kernel_size_w, int stride_d,
    int stride_h, int stride_w, int pad_d, int pad_h, int pad_w,
    const VecInt& out_shape, float* out_data, Context* context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_d = in_shape[2], in_h = in_shape[3], in_w = in_shape[4];
  int out_d = out_shape[2], out_h = out_shape[3], out_w = out_shape[4];
  int count = batch * channel * out_d * out_h * out_w;
  KernelPooling3D<<<GetBlocks(count), NumThreads, 0,
                    cudaStream_t(context->stream())>>>(
      in_data, count, channel, in_d, in_h, in_w, pool_type, kernel_size_d,
      kernel_size_h, kernel_size_w, stride_d, stride_h, stride_w, pad_d, pad_h,
      pad_w, out_d, out_h, out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PoolingGPU, PoolingKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class PoolingKernelCUDNN : public PoolingKernel {
 public:
  PoolingKernelCUDNN() {
    cudnn::createPoolingDesc<float>(&pooling_desc_);
    cudnn::createTensorDesc<float>(&in_desc_);
    cudnn::createTensorDesc<float>(&out_desc_);
  }
  ~PoolingKernelCUDNN() override {
    if (pooling_desc_ != nullptr) {
      cudnnDestroyPoolingDescriptor(pooling_desc_);
      pooling_desc_ = nullptr;
    }
    if (in_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(in_desc_);
      in_desc_ = nullptr;
    }
    if (out_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(out_desc_);
      out_desc_ = nullptr;
    }
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int pool_type, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w,
           bool full_pooling) override {
    int batch = input->shape(0), channel = input->shape(1);
    int in_h = input->shape(2), in_w = input->shape(3);
    int out_h = output->shape(2), out_w = output->shape(3);

    cudnn::setPooling2dDesc<float>(&pooling_desc_, pool_type, kernel_size_h,
                                   kernel_size_w, pad_h, pad_w, stride_h,
                                   stride_w);
    cudnn::setTensor4dDesc<float>(&in_desc_, batch, channel, in_h, in_w);
    cudnn::setTensor4dDesc<float>(&out_desc_, batch, channel, out_h, out_w);

    CUDNN_CHECK(cudnnPoolingForward(cudnnHandle_t(ws->Ctx()->cudnn_handle()),
                                    pooling_desc_, cudnn::dataType<float>::one,
                                    in_desc_, input->data<float>(),
                                    cudnn::dataType<float>::zero, out_desc_,
                                    output->mutable_data<float>()));
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int pool_type, int kernel_size_d, int kernel_size_h,
           int kernel_size_w, int stride_d, int stride_h, int stride_w,
           int pad_d, int pad_h, int pad_w, bool full_pooling) override {
    int num_axes = input->num_axes();

    const auto &in_shape = input->shape(), &out_shape = output->shape();

    std::array<int, 3> kernel_size{kernel_size_d, kernel_size_h, kernel_size_w};
    std::array<int, 3> stride{stride_d, stride_h, stride_w};
    std::array<int, 3> pad{pad_d, pad_h, pad_w};

    cudnn::setPoolingNdDesc<float>(&pooling_desc_, pool_type, 3,
                                   kernel_size.data(), pad.data(),
                                   stride.data());
    cudnn::setTensorNdDesc<float>(&in_desc_, num_axes, in_shape.data());
    cudnn::setTensorNdDesc<float>(&out_desc_, num_axes, out_shape.data());

    CUDNN_CHECK(cudnnPoolingForward(cudnnHandle_t(ws->Ctx()->cudnn_handle()),
                                    pooling_desc_, cudnn::dataType<float>::one,
                                    in_desc_, input->data<float>(),
                                    cudnn::dataType<float>::zero, out_desc_,
                                    output->mutable_data<float>()));
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnPoolingDescriptor_t pooling_desc_ = nullptr;
  cudnnTensorDescriptor_t in_desc_ = nullptr, out_desc_ = nullptr;
};

REGISTER_OP_KERNEL_CUDNN(PoolingGPU, PoolingKernelCUDNN);

#endif

}  // namespace Shadow
