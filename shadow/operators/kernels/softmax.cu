#include "softmax.hpp"

#include <cfloat>

namespace Shadow {

namespace Vision {

__global__ void KernelChannelMax(const float* in_data, int val_count, int dim,
                                 int inner_num, float* val_data) {
  CUDA_KERNEL_LOOP(globalid, val_count) {
    int n = globalid / inner_num, s = globalid % inner_num;
    const auto* in_data_offset = in_data + n * dim * inner_num + s;
    auto max_val = -FLT_MAX;
    for (int c = 0; c < dim; ++c, in_data_offset += inner_num) {
      max_val = fmaxf(*in_data_offset, max_val);
    }
    val_data[globalid] = max_val;
  }
}

__global__ void KernelChannelSubExp(const float* in_data, const float* val_data,
                                    int count, int dim, int inner_num,
                                    float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n = globalid / dim / inner_num, s = globalid % inner_num;
    out_data[globalid] = expf(in_data[globalid] - val_data[n * inner_num + s]);
  }
}

__global__ void KernelChannelSum(const float* out_data, int val_count, int dim,
                                 int inner_num, float* val_data) {
  CUDA_KERNEL_LOOP(globalid, val_count) {
    int n = globalid / inner_num, s = globalid % inner_num;
    const auto* out_data_offset = out_data + n * dim * inner_num + s;
    float sum = 0.f;
    for (int c = 0; c < dim; ++c, out_data_offset += inner_num) {
      sum += *out_data_offset;
    }
    val_data[globalid] = sum;
  }
}

__global__ void KernelChannelDiv(const float* val_data, int count, int dim,
                                 int inner_num, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n = globalid / dim / inner_num, s = globalid % inner_num;
    out_data[globalid] /= val_data[n * inner_num + s];
  }
}

template <>
void Softmax<DeviceType::kGPU, float>(const float* in_data, int outer_num,
                                      int dim, int inner_num, float* val_data,
                                      float* out_data, Context* context) {
  int val_count = outer_num * inner_num, count = val_count * dim;
  KernelChannelMax<<<GetBlocks(val_count), NumThreads, 0,
                     cudaStream_t(context->cuda_stream())>>>(
      in_data, val_count, dim, inner_num, val_data);
  KernelChannelSubExp<<<GetBlocks(count), NumThreads, 0,
                        cudaStream_t(context->cuda_stream())>>>(
      in_data, val_data, count, dim, inner_num, out_data);
  KernelChannelSum<<<GetBlocks(val_count), NumThreads, 0,
                     cudaStream_t(context->cuda_stream())>>>(
      out_data, val_count, dim, inner_num, val_data);
  KernelChannelDiv<<<GetBlocks(count), NumThreads, 0,
                     cudaStream_t(context->cuda_stream())>>>(
      val_data, count, dim, inner_num, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(SoftmaxGPU, SoftmaxKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class SoftmaxKernelCUDNN : public SoftmaxKernel {
 public:
  SoftmaxKernelCUDNN() { cudnn::createTensorDesc<float>(&in_out_desc_); }
  ~SoftmaxKernelCUDNN() override {
    if (in_out_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(in_out_desc_);
      in_out_desc_ = nullptr;
    }
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int axis) override {
    int outer_num = input->count(0, axis), dim = input->shape(axis),
        inner_num = input->count(axis + 1);

    cudnn::setTensor4dDesc<float>(&in_out_desc_, outer_num, dim, inner_num, 1);

    CUDNN_CHECK(cudnnSoftmaxForward(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL, cudnn::dataType<float>::one, in_out_desc_,
        input->data<float>(), cudnn::dataType<float>::zero, in_out_desc_,
        output->mutable_data<float>()));
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnTensorDescriptor_t in_out_desc_ = nullptr;
};

REGISTER_OP_KERNEL_CUDNN(SoftmaxGPU, SoftmaxKernelCUDNN);

#endif

}  // namespace Shadow
