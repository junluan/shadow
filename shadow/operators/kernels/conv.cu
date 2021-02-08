#include "conv.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelIm2Col2D(const float* in_data, int count, int in_h,
                               int in_w, int kernel_size_h, int kernel_size_w,
                               int stride_h, int stride_w, int pad_h, int pad_w,
                               int dilation_h, int dilation_w, int out_h,
                               int out_w, float* col_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    int c_in = temp / out_h;
    int c_out = c_in * kernel_size_h * kernel_size_w;
    int h_offset = h_out * stride_h - pad_h;
    int w_offset = w_out * stride_w - pad_w;
    in_data += (c_in * in_h + h_offset) * in_w + w_offset;
    col_data += (c_out * out_h + h_out) * out_w + w_out;
    for (int kh = 0; kh < kernel_size_h; ++kh) {
      int h_in = h_offset + kh * dilation_h;
      for (int kw = 0; kw < kernel_size_w; ++kw) {
        int w_in = w_offset + kw * dilation_w;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
          *col_data = in_data[kh * dilation_h * in_w + kw * dilation_w];
        } else {
          *col_data = 0.f;
        }
        col_data += out_h * out_w;
      }
    }
  }
}

__global__ void KernelIm2Col3D(const float* in_data, int count, int in_d,
                               int in_h, int in_w, int kernel_size_d,
                               int kernel_size_h, int kernel_size_w,
                               int stride_d, int stride_h, int stride_w,
                               int pad_d, int pad_h, int pad_w, int dilation_d,
                               int dilation_h, int dilation_w, int out_d,
                               int out_h, int out_w, float* col_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp /= out_h;
    int d_out = temp % out_d;
    int c_in = temp / out_d;
    int c_out = c_in * kernel_size_d * kernel_size_h * kernel_size_w;
    int d_offset = d_out * stride_d - pad_d;
    int h_offset = h_out * stride_h - pad_h;
    int w_offset = w_out * stride_w - pad_w;
    in_data += ((c_in * in_d + d_offset) * in_h + h_offset) * in_w + w_offset;
    col_data += ((c_out * out_d + d_out) * out_h + h_out) * out_w + w_out;
    for (int kd = 0; kd < kernel_size_d; ++kd) {
      int d_in = d_offset + kd * dilation_d;
      for (int kh = 0; kh < kernel_size_h; ++kh) {
        int h_in = h_offset + kh * dilation_h;
        for (int kw = 0; kw < kernel_size_w; ++kw) {
          int w_in = w_offset + kw * dilation_w;
          if (d_in >= 0 && d_in < in_d && h_in >= 0 && h_in < in_h &&
              w_in >= 0 && w_in < in_w) {
            *col_data =
                in_data[(kd * dilation_d * in_h + kh * dilation_h) * in_w +
                        kw * dilation_w];
          } else {
            *col_data = 0.f;
          }
          col_data += out_d * out_h * out_w;
        }
      }
    }
  }
}

template <>
void Im2Col2D<DeviceType::kGPU, float>(const float* in_data,
                                       const VecInt& in_shape,
                                       int kernel_size_h, int kernel_size_w,
                                       int stride_h, int stride_w, int pad_h,
                                       int pad_w, int dilation_h,
                                       int dilation_w, const VecInt& out_shape,
                                       float* col_data, Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * out_h * out_w;
  KernelIm2Col2D<<<GetBlocks(count), NumThreads, 0,
                   cudaStream_t(context->stream())>>>(
      in_data, count, in_h, in_w, kernel_size_h, kernel_size_w, stride_h,
      stride_w, pad_h, pad_w, dilation_h, dilation_w, out_h, out_w, col_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <>
void Im2Col3D<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, int kernel_size_d,
    int kernel_size_h, int kernel_size_w, int stride_d, int stride_h,
    int stride_w, int pad_d, int pad_h, int pad_w, int dilation_d,
    int dilation_h, int dilation_w, const VecInt& out_shape, float* col_data,
    Context* context) {
  int in_c = in_shape[1], in_d = in_shape[2], in_h = in_shape[3],
      in_w = in_shape[4];
  int out_d = out_shape[2], out_h = out_shape[3], out_w = out_shape[4];
  int count = in_c * out_d * out_h * out_w;
  KernelIm2Col3D<<<GetBlocks(count), NumThreads, 0,
                   cudaStream_t(context->stream())>>>(
      in_data, count, in_d, in_h, in_w, kernel_size_d, kernel_size_h,
      kernel_size_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
      dilation_d, dilation_h, dilation_w, out_d, out_h, out_w, col_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void KernelDepthwise2D(const float* in_data, int count,
                                  const float* weight_data,
                                  const float* bias_data, int in_c, int in_h,
                                  int in_w, int out_c, int out_h, int out_w,
                                  int kernel_size_h, int kernel_size_w,
                                  int stride_h, int stride_w, int pad_h,
                                  int pad_w, int dilation_h, int dilation_w,
                                  bool bias_term, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp /= out_h;
    int c_out = temp % out_c;
    int b_out = temp / out_c;

    in_data += (b_out * in_c + c_out / (out_c / in_c)) * in_h * in_w;
    weight_data += c_out * kernel_size_h * kernel_size_w;

    double sum_val = 0;
    for (int kh = 0; kh < kernel_size_h; ++kh) {
      int h_in = h_out * stride_h - pad_h + kh * dilation_h;
      for (int kw = 0; kw < kernel_size_w; ++kw) {
        int w_in = w_out * stride_w - pad_w + kw * dilation_w;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
          sum_val += in_data[h_in * in_w + w_in] * *weight_data;
        }
        weight_data++;
      }
    }
    if (bias_term) {
      sum_val += bias_data[c_out];
    }

    out_data[globalid] = static_cast<float>(sum_val);
  }
}

__global__ void KernelDepthwise3D(
    const float* in_data, int count, const float* weight_data,
    const float* bias_data, int in_c, int in_d, int in_h, int in_w, int out_c,
    int out_d, int out_h, int out_w, int kernel_size_d, int kernel_size_h,
    int kernel_size_w, int stride_d, int stride_h, int stride_w, int pad_d,
    int pad_h, int pad_w, int dilation_d, int dilation_h, int dilation_w,
    bool bias_term, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp /= out_h;
    int d_out = temp % out_d;
    temp /= out_d;
    int c_out = temp % out_c;
    int b_out = temp / out_c;

    in_data += (b_out * in_c + c_out / (out_c / in_c)) * in_d * in_h * in_w;
    weight_data += c_out * kernel_size_d * kernel_size_h * kernel_size_w;

    double sum_val = 0;
    for (int kd = 0; kd < kernel_size_d; ++kd) {
      int d_in = d_out * stride_d - pad_d + kd * dilation_d;
      for (int kh = 0; kh < kernel_size_h; ++kh) {
        int h_in = h_out * stride_h - pad_h + kh * dilation_h;
        for (int kw = 0; kw < kernel_size_w; ++kw) {
          int w_in = w_out * stride_w - pad_w + kw * dilation_w;
          if (d_in >= 0 && d_in < in_d && h_in >= 0 && h_in < in_h &&
              w_in >= 0 && w_in < in_w) {
            sum_val +=
                in_data[(d_in * in_h + h_in) * in_w + w_in] * *weight_data;
          }
          weight_data++;
        }
      }
    }
    if (bias_term) {
      sum_val += bias_data[c_out];
    }

    out_data[globalid] = static_cast<float>(sum_val);
  }
}

template <>
void Depthwise2D<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, const float* weight_data,
    const float* bias_data, int kernel_size_h, int kernel_size_w, int stride_h,
    int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w,
    bool bias_term, const VecInt& out_shape, float* out_data,
    Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_c = out_shape[1], out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * out_c * out_h * out_w;
  KernelDepthwise2D<<<GetBlocks(count), NumThreads, 0,
                      cudaStream_t(context->stream())>>>(
      in_data, count, weight_data, bias_data, in_c, in_h, in_w, out_c, out_h,
      out_w, kernel_size_h, kernel_size_w, stride_h, stride_w, pad_h, pad_w,
      dilation_h, dilation_w, bias_term, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <>
void Depthwise3D<DeviceType::kGPU, float>(
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
  int count = batch * out_c * out_d * out_h * out_w;
  KernelDepthwise3D<<<GetBlocks(count), NumThreads, 0,
                      cudaStream_t(context->stream())>>>(
      in_data, count, weight_data, bias_data, in_c, in_d, in_h, in_w, out_c,
      out_d, out_h, out_w, kernel_size_d, kernel_size_h, kernel_size_w,
      stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h,
      dilation_w, bias_term, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ConvGPU, ConvKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class ConvKernelCUDNN : public ConvKernel {
 public:
  ConvKernelCUDNN() {
    cudnn::createConvolutionDesc<float>(&conv_desc_);
    cudnn::createTensorDesc<float>(&in_desc_);
    cudnn::createTensorDesc<float>(&out_desc_);
    cudnn::createFilterDesc<float>(&weight_desc_);
    cudnn::createTensorDesc<float>(&bias_desc_);
    cudnn::createActivationDesc<float>(&activate_desc_);
  }
  ~ConvKernelCUDNN() override {
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
           int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
           int dilation_w, int group, bool bias_term,
           int activate_type) override {
    int batch = input->shape(0), in_c = input->shape(1), in_h = input->shape(2),
        in_w = input->shape(3);
    int out_h = output->shape(2), out_w = output->shape(3);

    cudnn::setConvolution2dDesc<float>(&conv_desc_, pad_h, pad_w, stride_h,
                                       stride_w, dilation_h, dilation_w, group);
    cudnn::setTensor4dDesc<float>(&in_desc_, batch, in_c, in_h, in_w);
    cudnn::setTensor4dDesc<float>(&out_desc_, batch, num_output, out_h, out_w);
    cudnn::setFilter4dDesc<float>(&weight_desc_, num_output, in_c / group,
                                  kernel_size_h, kernel_size_w);
    if (bias_term) {
      cudnn::setTensor4dDesc<float>(&bias_desc_, 1, num_output, 1, 1);
    }
    if (activate_type == 1) {
      cudnn::setActivationDesc<float>(&activate_desc_, activate_type, 0);
    }

#if !CUDNN_VERSION_MIN(8, 0, 0)
    size_t workspace_limit_bytes = group == 1 ? 64 * 1024 * 1024 : 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), in_desc_, weight_desc_,
        conv_desc_, out_desc_, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &fwd_algo_));
#endif

    size_t workspace_fwd_size = 0;

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), in_desc_, weight_desc_,
        conv_desc_, out_desc_, fwd_algo_, &workspace_fwd_size));

    std::shared_ptr<Blob> workspace = nullptr;
    const void* workspace_ptr = nullptr;
    if (workspace_fwd_size > 0) {
      ws->GrowTempBuffer(workspace_fwd_size);
      workspace = ws->CreateTempBlob({static_cast<int>(workspace_fwd_size)},
                                     DataType::kU8);
      workspace_ptr = workspace->data<unsigned char>();
    }

    CUDNN_CHECK(cudnnConvolutionForward(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), cudnn::dataType<float>::one,
        in_desc_, input->data<float>(), weight_desc_, weight->data<float>(),
        conv_desc_, fwd_algo_, const_cast<void*>(workspace_ptr),
        workspace_fwd_size, cudnn::dataType<float>::zero, out_desc_,
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

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, int kernel_size_d, int kernel_size_h,
           int kernel_size_w, int stride_d, int stride_h, int stride_w,
           int pad_d, int pad_h, int pad_w, int dilation_d, int dilation_h,
           int dilation_w, int group, bool bias_term,
           int activate_type) override {
    int num_axes = input->num_axes();

    const auto &in_shape = input->shape(), &weight_shape = weight->shape(),
               &out_shape = output->shape();

    std::array<int, 3> stride{stride_d, stride_h, stride_w};
    std::array<int, 3> pad{pad_d, pad_h, pad_w};
    std::array<int, 3> dilation{dilation_d, dilation_h, dilation_w};

    cudnn::setConvolutionNdDesc<float>(&conv_desc_, 3, pad.data(),
                                       stride.data(), dilation.data(), group);
    cudnn::setTensorNdDesc<float>(&in_desc_, num_axes, in_shape.data());
    cudnn::setTensorNdDesc<float>(&out_desc_, num_axes, out_shape.data());
    cudnn::setFilterNdDesc<float>(&weight_desc_, weight->num_axes(),
                                  weight_shape.data());
    if (bias_term) {
      std::array<int, 5> bias_shape{1, num_output, 1, 1, 1};
      cudnn::setTensorNdDesc<float>(&bias_desc_, num_axes, bias_shape.data());
    }
    if (activate_type == 1) {
      cudnn::setActivationDesc<float>(&activate_desc_, activate_type, 0);
    }

#if !CUDNN_VERSION_MIN(8, 0, 0)
    size_t workspace_limit_bytes = group == 1 ? 64 * 1024 * 1024 : 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), in_desc_, weight_desc_,
        conv_desc_, out_desc_, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &fwd_algo_));
#endif

    size_t workspace_fwd_size = 0;

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), in_desc_, weight_desc_,
        conv_desc_, out_desc_, fwd_algo_, &workspace_fwd_size));

    std::shared_ptr<Blob> workspace = nullptr;
    const void* workspace_ptr = nullptr;
    if (workspace_fwd_size > 0) {
      ws->GrowTempBuffer(workspace_fwd_size);
      workspace = ws->CreateTempBlob({static_cast<int>(workspace_fwd_size)},
                                     DataType::kU8);
      workspace_ptr = workspace->data<unsigned char>();
    }

    CUDNN_CHECK(cudnnConvolutionForward(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), cudnn::dataType<float>::one,
        in_desc_, input->data<float>(), weight_desc_, weight->data<float>(),
        conv_desc_, fwd_algo_, const_cast<void*>(workspace_ptr),
        workspace_fwd_size, cudnn::dataType<float>::zero, out_desc_,
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
  cudnnConvolutionFwdAlgo_t fwd_algo_ =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
  cudnnTensorDescriptor_t in_desc_ = nullptr, out_desc_ = nullptr;
  cudnnFilterDescriptor_t weight_desc_ = nullptr;
  cudnnTensorDescriptor_t bias_desc_ = nullptr;

  cudnnActivationDescriptor_t activate_desc_ = nullptr;
};

REGISTER_OP_KERNEL_CUDNN(ConvGPU, ConvKernelCUDNN);

#endif

}  // namespace Shadow
