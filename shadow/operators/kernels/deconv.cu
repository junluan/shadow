#include "deconv.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelCol2Im(const float* col_data, int offset, int count,
                             int in_c, int in_h, int in_w, int kernel_size_h,
                             int kernel_size_w, int stride_h, int stride_w,
                             int pad_h, int pad_w, int dilation, int out_h,
                             int out_w, float* in_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int w_im = globalid % in_w + pad_w;
    int h_im = (globalid / in_w) % in_h + pad_h;
    int c_im = globalid / (in_w * in_h);
    int kernel_extent_h = (kernel_size_h - 1) * dilation + 1;
    int kernel_extent_w = (kernel_size_w - 1) * dilation + 1;
    int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    int w_col_end = min(w_im / stride_w + 1, out_w);
    int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    int h_col_end = min(h_im / stride_h + 1, out_h);
    double val = 0;
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation == 0 && w_k % dilation == 0) {
          h_k /= dilation, w_k /= dilation;
          int col_index =
              (((c_im * kernel_size_h + h_k) * kernel_size_w + w_k) * out_h +
               h_col) *
                  out_w +
              w_col;
          val += col_data[col_index];
        }
      }
    }
    in_data[globalid + offset] = static_cast<float>(val);
  }
}

template <>
void Col2Im<DeviceType::kGPU, float>(const float* col_data,
                                     const VecInt& in_shape, int offset,
                                     int kernel_size_h, int kernel_size_w,
                                     int stride_h, int stride_w, int pad_h,
                                     int pad_w, int dilation,
                                     const VecInt& out_shape, float* in_data,
                                     Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * in_h * in_w;
  KernelCol2Im<<<GetBlocks(count), NumThreads, 0,
                 cudaStream_t(context->cuda_stream())>>>(
      col_data, offset, count, in_c, in_h, in_w, kernel_size_h, kernel_size_w,
      stride_h, stride_w, pad_h, pad_w, dilation, out_h, out_w, in_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DeconvGPU, DeconvKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class DeconvKernelCUDNN : public DeconvKernel {
 public:
  DeconvKernelCUDNN() {
    cudnn::createConvolutionDesc<float>(&conv_desc_);
    cudnn::createTensorDesc<float>(&in_desc_);
    cudnn::createTensorDesc<float>(&out_desc_);
    cudnn::createFilterDesc<float>(&weight_desc_);
    cudnn::createTensorDesc<float>(&bias_desc_);
    cudnn::createActivationDesc<float>(&activate_desc_);
  }
  ~DeconvKernelCUDNN() override {
    if (conv_desc_ != nullptr) {
      cudnnDestroyConvolutionDescriptor(conv_desc_);
      conv_desc_ = nullptr;
    }
    if (in_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(in_desc_);
      in_desc_ = nullptr;
    }
    if (out_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(out_desc_);
      out_desc_ = nullptr;
    }
    if (weight_desc_ != nullptr) {
      cudnnDestroyFilterDescriptor(weight_desc_);
      weight_desc_ = nullptr;
    }
    if (bias_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bias_desc_);
      bias_desc_ = nullptr;
    }
    if (activate_desc_ != nullptr) {
      cudnnDestroyActivationDescriptor(activate_desc_);
      activate_desc_ = nullptr;
    }
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation,
           int group, bool bias_term, int activate_type) override {
    int batch = input->shape(0), in_c = input->shape(1), in_h = input->shape(2),
        in_w = input->shape(3);
    int out_h = output->shape(2), out_w = output->shape(3);

    cudnn::setConvolution2dDesc<float>(&conv_desc_, pad_h, pad_w, stride_h,
                                       stride_w, dilation, dilation, group);
    cudnn::setTensor4dDesc<float>(&in_desc_, batch, in_c, in_h, in_w);
    cudnn::setTensor4dDesc<float>(&out_desc_, batch, num_output, out_h, out_w);
    cudnn::setFilter4dDesc<float>(&weight_desc_, in_c, num_output / group,
                                  kernel_size_h, kernel_size_w);
    if (bias_term) {
      cudnn::setTensor4dDesc<float>(&bias_desc_, 1, num_output, 1, 1);
    }
    if (activate_type == 1) {
      cudnn::setActivationDesc<float>(&activate_desc_, activate_type, 0);
    }

    size_t workspace_limit_bytes = group == 1 ? 64 * 1024 * 1024 : 0;

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), weight_desc_, in_desc_,
        conv_desc_, out_desc_,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_));

    size_t workspace_bwd_size = 0;

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), weight_desc_, in_desc_,
        conv_desc_, out_desc_, bwd_data_algo_, &workspace_bwd_size));

    std::shared_ptr<Blob> workspace = nullptr;
    const void* workspace_ptr = nullptr;
    if (workspace_bwd_size > 0) {
      ws->GrowTempBuffer(workspace_bwd_size);
      workspace = ws->CreateTempBlob({static_cast<int>(workspace_bwd_size)},
                                     DataType::kU8);
      workspace_ptr = workspace->data<unsigned char>();
    }

    CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), cudnn::dataType<float>::one,
        weight_desc_, weight->data<float>(), in_desc_, input->data<float>(),
        conv_desc_, bwd_data_algo_, const_cast<void*>(workspace_ptr),
        workspace_bwd_size, cudnn::dataType<float>::zero, out_desc_,
        output->mutable_data<float>()));

    if (bias_term) {
      CUDNN_CHECK(cudnnAddTensor(
          cudnnHandle_t(ws->Ctx()->cudnn_handle()), cudnn::dataType<float>::one,
          bias_desc_, bias->data<float>(), cudnn::dataType<float>::one,
          out_desc_, output->mutable_data<float>()));
    }

    if (activate_type == 1) {
      CUDNN_CHECK(cudnnActivationForward(
          cudnnHandle_t(ws->Ctx()->cudnn_handle()), activate_desc_,
          cudnn::dataType<float>::one, out_desc_, output->data<float>(),
          cudnn::dataType<float>::zero, out_desc_,
          output->mutable_data<float>()));
    }
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_ =
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

  cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
  cudnnTensorDescriptor_t in_desc_ = nullptr, out_desc_ = nullptr;
  cudnnFilterDescriptor_t weight_desc_ = nullptr;
  cudnnTensorDescriptor_t bias_desc_ = nullptr;

  cudnnActivationDescriptor_t activate_desc_ = nullptr;
};

REGISTER_OP_KERNEL_CUDNN(DeconvGPU, DeconvKernelCUDNN);

#endif

}  // namespace Shadow
