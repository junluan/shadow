#include "grid_sample.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelGridSampleNearest(const float* in_data,
                                        const float* grid_data, int count,
                                        int channel, int in_h, int in_w,
                                        int out_h, int out_w, int padding_mode,
                                        float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    int grid_offset = ((b_out * out_h + h_out) * out_w + w_out) * 2;
    float x = grid_data[grid_offset], y = grid_data[grid_offset + 1];

    int src_h = static_cast<int>(roundf((y + 1) / 2.f * (in_h - 1)));
    int src_w = static_cast<int>(roundf((x + 1) / 2.f * (in_w - 1)));

    if (padding_mode == 1) {
      src_h = min(max(src_h, 0), in_h - 1);
      src_w = min(max(src_w, 0), in_w - 1);
    } else if (padding_mode == 0) {
      if (src_h < 0 || src_w < 0 || src_h > in_h - 1 || src_w > in_w - 1) {
        *out_data++ = 0.f;
        continue;
      }
    }

    int src_index = ((b_out * channel + c_out) * in_h + src_h) * in_w + src_w;
    out_data[globalid] = in_data[src_index];
  }
}

__global__ void KernelGridSampleBilinear(const float* in_data,
                                         const float* grid_data, int count,
                                         int channel, int in_h, int in_w,
                                         int out_h, int out_w, int padding_mode,
                                         float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    int grid_offset = ((b_out * out_h + h_out) * out_w + w_out) * 2;
    float x = grid_data[grid_offset], y = grid_data[grid_offset + 1];

    float src_h_f = (y + 1) / 2.f * (in_h - 1);
    float src_w_f = (x + 1) / 2.f * (in_w - 1);

    if (padding_mode == 1) {
      src_h_f = fminf(fmaxf(src_h_f, 0.f), in_h - 1.f);
      src_w_f = fminf(fmaxf(src_w_f, 0.f), in_w - 1.f);
    } else if (padding_mode == 0) {
      if (src_h_f < 0 || src_w_f < 0 || src_h_f > in_h - 1 ||
          src_w_f > in_w - 1) {
        *out_data++ = 0.f;
        continue;
      }
    }

    int src_h_0 = static_cast<int>(fmaxf(floorf(src_h_f), 0.f));
    int src_h_1 = static_cast<int>(fminf(ceilf(src_h_f), in_h - 1.f));
    int src_w_0 = static_cast<int>(fmaxf(floorf(src_w_f), 0.f));
    int src_w_1 = static_cast<int>(fminf(ceilf(src_w_f), in_w - 1.f));
    float sh = src_h_f - src_h_0, sw = src_w_f - src_w_0;

    int h_offset = (b_out * channel + c_out) * in_h;
    int src_index_0 = (h_offset + src_h_0) * in_w + src_w_0;
    int src_index_1 = (h_offset + src_h_1) * in_w + src_w_0;
    int src_index_2 = (h_offset + src_h_0) * in_w + src_w_1;
    int src_index_3 = (h_offset + src_h_1) * in_w + src_w_1;

    out_data[globalid] = (1 - sh) * (1 - sw) * in_data[src_index_0] +
                         sh * (1 - sw) * in_data[src_index_1] +
                         (1 - sh) * sw * in_data[src_index_2] +
                         sh * sw * in_data[src_index_3];
  }
}

template <>
void GridSample<DeviceType::kGPU, float>(const float* in_data,
                                         const VecInt& in_shape,
                                         const float* grid_data, int mode,
                                         int padding_mode,
                                         const VecInt& out_shape,
                                         float* out_data, Context* context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * channel * out_h * out_w;
  if (mode == 0) {
    KernelGridSampleNearest<<<GetBlocks(count), NumThreads, 0,
                              cudaStream_t(context->stream())>>>(
        in_data, grid_data, count, channel, in_h, in_w, out_h, out_w,
        padding_mode, out_data);
  } else if (mode == 1) {
    KernelGridSampleBilinear<<<GetBlocks(count), NumThreads, 0,
                               cudaStream_t(context->stream())>>>(
        in_data, grid_data, count, channel, in_h, in_w, out_h, out_w,
        padding_mode, out_data);
  } else {
    LOG(FATAL) << "Unsupported grid sample mode: " << mode;
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(GridSampleGPU,
                           GridSampleKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class GridSampleKernelCUDNN : public GridSampleKernel {
 public:
  GridSampleKernelCUDNN() {
    cudnn::createSpatialTransformerDesc<float>(&spatial_transformer_desc_);
    cudnn::createTensorDesc<float>(&in_desc_);
    cudnn::createTensorDesc<float>(&out_desc_);
    default_kernel_ =
        std::make_shared<GridSampleKernelDefault<DeviceType::kGPU>>();
  }
  ~GridSampleKernelCUDNN() override {
    if (spatial_transformer_desc_ != nullptr) {
      cudnnDestroySpatialTransformerDescriptor(spatial_transformer_desc_);
      spatial_transformer_desc_ = nullptr;
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

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& grid, std::shared_ptr<Blob>& output,
           Workspace* ws, int mode, int padding_mode) override {
    if (mode == 1 && padding_mode == 0) {
      const auto& in_shape = input->shape();
      int batch = input->shape(0), channel = input->shape(1),
          num_axes = input->num_axes();
      int out_h = grid->shape(1), out_w = grid->shape(2);

      cudnn::setSpatialTransformerDesc<float>(&spatial_transformer_desc_,
                                              num_axes, in_shape.data());
      cudnn::setTensorNdDesc<float>(&in_desc_, num_axes, in_shape.data());
      cudnn::setTensor4dDesc<float>(&out_desc_, batch, channel, out_h, out_w);

      CUDNN_CHECK(cudnnSpatialTfSamplerForward(
          cudnnHandle_t(ws->Ctx()->cudnn_handle()), spatial_transformer_desc_,
          cudnn::dataType<float>::one, in_desc_, input->data<float>(),
          grid->data<float>(), cudnn::dataType<float>::zero, out_desc_,
          output->mutable_data<float>()));
    } else {
      default_kernel_->Run(input, grid, output, ws, mode, padding_mode);
    }
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnSpatialTransformerDescriptor_t spatial_transformer_desc_ = nullptr;
  cudnnTensorDescriptor_t in_desc_ = nullptr, out_desc_ = nullptr;

  std::shared_ptr<GridSampleKernelDefault<DeviceType::kGPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_CUDNN(GridSampleGPU, GridSampleKernelCUDNN);

#endif

}  // namespace Shadow
