#include "deconv.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelCol2Im2D(const float* col_data, int count, int in_h,
                               int in_w, int kernel_size_h, int kernel_size_w,
                               int stride_h, int stride_w, int pad_h, int pad_w,
                               int dilation_h, int dilation_w, int out_h,
                               int out_w, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    int c_out = temp / out_h;
    h_out += pad_h, w_out += pad_w;
    int ke_h = (kernel_size_h - 1) * dilation_h + 1;
    int ke_w = (kernel_size_w - 1) * dilation_w + 1;
    int hstart = h_out < ke_h ? 0 : (h_out - ke_h) / stride_h + 1;
    int hend = min(h_out / stride_h + 1, in_h);
    int wstart = w_out < ke_w ? 0 : (w_out - ke_w) / stride_w + 1;
    int wend = min(w_out / stride_w + 1, in_w);
    col_data += in_h * in_w * c_out * kernel_size_h * kernel_size_w;
    double sum_val = 0;
    for (int h_in = hstart; h_in < hend; ++h_in) {
      int h_k = h_out - h_in * stride_h;
      if (h_k % dilation_h == 0) {
        for (int w_in = wstart; w_in < wend; ++w_in) {
          int w_k = w_out - w_in * stride_w;
          if (w_k % dilation_w == 0) {
            sum_val += col_data
                [((h_k / dilation_h * kernel_size_w + w_k / dilation_w) * in_h +
                  h_in) *
                     in_w +
                 w_in];
          }
        }
      }
    }
    out_data[globalid] = static_cast<float>(sum_val);
  }
}

__global__ void KernelCol2Im3D(const float* col_data, int count, int in_d,
                               int in_h, int in_w, int kernel_size_d,
                               int kernel_size_h, int kernel_size_w,
                               int stride_d, int stride_h, int stride_w,
                               int pad_d, int pad_h, int pad_w, int dilation_d,
                               int dilation_h, int dilation_w, int out_d,
                               int out_h, int out_w, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp /= out_h;
    int d_out = temp % out_d;
    int c_out = temp / out_d;
    d_out += pad_d, h_out += pad_h, w_out += pad_w;
    int ke_d = (kernel_size_d - 1) * dilation_d + 1;
    int ke_h = (kernel_size_h - 1) * dilation_h + 1;
    int ke_w = (kernel_size_w - 1) * dilation_w + 1;
    int dstart = d_out < ke_d ? 0 : (d_out - ke_d) / stride_d + 1;
    int dend = min(d_out / stride_d + 1, in_d);
    int hstart = h_out < ke_h ? 0 : (h_out - ke_h) / stride_h + 1;
    int hend = min(h_out / stride_h + 1, in_h);
    int wstart = w_out < ke_w ? 0 : (w_out - ke_w) / stride_w + 1;
    int wend = min(w_out / stride_w + 1, in_w);
    col_data += in_d * in_h * in_w * c_out * kernel_size_d * kernel_size_h *
                kernel_size_w;
    double sum_val = 0;
    for (int d_in = dstart; d_in < dend; ++d_in) {
      int d_k = d_out - d_in * stride_d;
      if (d_k % dilation_d == 0) {
        for (int h_in = hstart; h_in < hend; ++h_in) {
          int h_k = h_out - h_in * stride_h;
          if (h_k % dilation_h == 0) {
            for (int w_in = wstart; w_in < wend; ++w_in) {
              int w_k = w_out - w_in * stride_w;
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
    out_data[globalid] = static_cast<float>(sum_val);
  }
}

template <>
void Col2Im2D<DeviceType::kGPU, float>(const float* col_data,
                                       const VecInt& in_shape,
                                       int kernel_size_h, int kernel_size_w,
                                       int stride_h, int stride_w, int pad_h,
                                       int pad_w, int dilation_h,
                                       int dilation_w, const VecInt& out_shape,
                                       float* out_data, Context* context) {
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_c = out_shape[1], out_h = out_shape[2], out_w = out_shape[3];
  int count = out_c * out_h * out_w;
  KernelCol2Im2D<<<GetBlocks(count), NumThreads, 0,
                   cudaStream_t(context->stream())>>>(
      col_data, count, in_h, in_w, kernel_size_h, kernel_size_w, stride_h,
      stride_w, pad_h, pad_w, dilation_h, dilation_w, out_h, out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <>
void Col2Im3D<DeviceType::kGPU, float>(
    const float* col_data, const VecInt& in_shape, int kernel_size_d,
    int kernel_size_h, int kernel_size_w, int stride_d, int stride_h,
    int stride_w, int pad_d, int pad_h, int pad_w, int dilation_d,
    int dilation_h, int dilation_w, const VecInt& out_shape, float* out_data,
    Context* context) {
  int in_d = in_shape[2], in_h = in_shape[3], in_w = in_shape[4];
  int out_c = out_shape[1], out_d = out_shape[2], out_h = out_shape[3],
      out_w = out_shape[4];
  int count = out_c * out_d * out_h * out_w;
  KernelCol2Im3D<<<GetBlocks(count), NumThreads, 0,
                   cudaStream_t(context->stream())>>>(
      col_data, count, in_d, in_h, in_w, kernel_size_d, kernel_size_h,
      kernel_size_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
      dilation_d, dilation_h, dilation_w, out_d, out_h, out_w, out_data);
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
    cudnn::setFilter4dDesc<float>(&weight_desc_, in_c, num_output / group,
                                  kernel_size_h, kernel_size_w);
    if (bias_term) {
      cudnn::setTensor4dDesc<float>(&bias_desc_, 1, num_output, 1, 1);
    }
    if (activate_type == 1) {
      cudnn::setActivationDesc<float>(&activate_desc_, activate_type, 0);
    }

#if !CUDNN_VERSION_MIN(8, 0, 0)
    size_t workspace_limit_bytes = group == 1 ? 64 * 1024 * 1024 : 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), weight_desc_, in_desc_,
        conv_desc_, out_desc_,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_));
#endif

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
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), weight_desc_, in_desc_,
        conv_desc_, out_desc_,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_));
#endif

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
