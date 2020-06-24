#include "pooling.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelPooling(const float* in_data, int count, int in_c,
                              int in_h, int in_w, int pool_type,
                              int kernel_size_h, int kernel_size_w,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int out_h, int out_w, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int j_out = globalid % out_w;
    int i_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % in_c;
    int b_out = temp / in_c;

    int kistart = i_out * stride_h - pad_h, kjstart = j_out * stride_w - pad_w;
    int kiend = min(kistart + kernel_size_h, in_h + pad_h);
    int kjend = min(kjstart + kernel_size_w, in_w + pad_w);
    int pool_size = (kiend - kistart) * (kjend - kjstart);
    kistart = max(kistart, 0), kjstart = max(kjstart, 0);
    kiend = min(kiend, in_h), kjend = min(kjend, in_w);

    in_data += (b_out * in_c + c_out) * in_h * in_w;

    auto max_val = -FLT_MAX, sum_val = 0.f;
    for (int ki = kistart; ki < kiend; ++ki) {
      for (int kj = kjstart; kj < kjend; ++kj) {
        auto value = in_data[ki * in_w + kj];
        max_val = fmaxf(max_val, value);
        sum_val += value;
      }
    }
    out_data[globalid] = (pool_type == 0) ? max_val : sum_val / pool_size;
  }
}

template <>
void Pooling<DeviceType::kGPU, float>(const float* in_data,
                                      const VecInt& in_shape, int pool_type,
                                      int kernel_size_h, int kernel_size_w,
                                      int stride_h, int stride_w, int pad_h,
                                      int pad_w, const VecInt& out_shape,
                                      float* out_data, Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelPooling<<<GetBlocks(count), NumThreads, 0,
                  cudaStream_t(context->cuda_stream())>>>(
      in_data, count, in_c, in_h, in_w, pool_type, kernel_size_h, kernel_size_w,
      stride_h, stride_w, pad_h, pad_w, out_h, out_w, out_data);
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
    int batch = input->shape(0), in_c = input->shape(1);
    int in_h = input->shape(2), in_w = input->shape(3);
    int out_h = output->shape(2), out_w = output->shape(3);

    cudnn::setPooling2dDesc<float>(&pooling_desc_, pool_type, kernel_size_h,
                                   kernel_size_w, pad_h, pad_w, stride_h,
                                   stride_w);
    cudnn::setTensor4dDesc<float>(&in_desc_, batch, in_c, in_h, in_w);
    cudnn::setTensor4dDesc<float>(&out_desc_, batch, in_c, out_h, out_w);

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
