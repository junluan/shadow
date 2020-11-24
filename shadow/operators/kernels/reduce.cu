#include "reduce.hpp"

namespace Shadow {

namespace Vision {

__device__ float Reduce(const float* data, const int* list, int num_list,
                        int offset, int operation) {
  switch (operation) {
    case kProd: {
      double val = 1;
      for (int i = 0; i < num_list; ++i) {
        val *= data[list[i] + offset];
      }
      return static_cast<float>(val);
    }
    case kSum: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += data[list[i] + offset];
      }
      return static_cast<float>(val);
    }
    case kMax: {
      float val = -FLT_MAX;
      for (int i = 0; i < num_list; ++i) {
        val = fmaxf(val, data[list[i] + offset]);
      }
      return val;
    }
    case kMin: {
      float val = FLT_MAX;
      for (int i = 0; i < num_list; ++i) {
        val = fminf(val, data[list[i] + offset]);
      }
      return val;
    }
    case kAvg: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += data[list[i] + offset];
      }
      return static_cast<float>(val / num_list);
    }
    case kLpNorm1: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += fabsf(data[list[i] + offset]);
      }
      return static_cast<float>(val);
    }
    case kLpNorm2: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        auto abs_data = fabsf(data[list[i] + offset]);
        val += abs_data * abs_data;
      }
      return sqrtf(static_cast<float>(val));
    }
    default:
      return 0;
  }
}

__global__ void KernelReduce(const float* in_data, const int* list_data,
                             const int* offset_data, int num_list,
                             int operation, int count, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] =
        Reduce(in_data, list_data, num_list, offset_data[globalid], operation);
  }
}

template <>
void Reduce<DeviceType::kGPU, float>(const float* in_data, const int* list_data,
                                     const int* offset_data, int num_list,
                                     int operation, int count, float* out_data,
                                     Context* context) {
  KernelReduce<<<GetBlocks(count), NumThreads, 0,
                 cudaStream_t(context->stream())>>>(
      in_data, list_data, offset_data, num_list, operation, count, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ReduceGPU, ReduceKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class ReduceKernelCUDNN : public ReduceKernel {
 public:
  ReduceKernelCUDNN() {
    cudnn::createReduceDesc<float>(&reduce_desc_);
    cudnn::createTensorDesc<float>(&in_desc_);
    cudnn::createTensorDesc<float>(&out_desc_);
  }
  ~ReduceKernelCUDNN() override {
    if (reduce_desc_ != nullptr) {
      cudnnDestroyReduceTensorDescriptor(reduce_desc_);
      reduce_desc_ = nullptr;
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
           Workspace* ws, int operation, const VecInt& axes) override {
    int num_axes = input->num_axes();

    auto in_shape = input->shape(), out_shape = output->shape();

    cudnn::setReduceDesc<float>(&reduce_desc_, operation);
    if (num_axes > 4) {
      cudnn::setTensorNdDesc<float>(&in_desc_, num_axes, in_shape.data());
      cudnn::setTensorNdDesc<float>(&out_desc_, num_axes, out_shape.data());
    } else {
      in_shape.insert(in_shape.end(), 4 - num_axes, 1);
      out_shape.insert(out_shape.end(), 4 - num_axes, 1);
      cudnn::setTensor4dDesc<float>(&in_desc_, in_shape[0], in_shape[1],
                                    in_shape[2], in_shape[3]);
      cudnn::setTensor4dDesc<float>(&out_desc_, out_shape[0], out_shape[1],
                                    out_shape[2], out_shape[3]);
    }

    size_t workspace_size = 0;

    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), reduce_desc_, in_desc_,
        out_desc_, &workspace_size));

    std::shared_ptr<Blob> workspace = nullptr;
    const void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
      ws->GrowTempBuffer(workspace_size);
      workspace =
          ws->CreateTempBlob({static_cast<int>(workspace_size)}, DataType::kU8);
      workspace_ptr = workspace->data<unsigned char>();
    }

    CUDNN_CHECK(cudnnReduceTensor(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), reduce_desc_, nullptr, 0,
        const_cast<void*>(workspace_ptr), workspace_size,
        cudnn::dataType<float>::one, in_desc_, input->data<float>(),
        cudnn::dataType<float>::zero, out_desc_,
        output->mutable_data<float>()));
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnReduceTensorDescriptor_t reduce_desc_ = nullptr;
  cudnnTensorDescriptor_t in_desc_ = nullptr, out_desc_ = nullptr;
};

REGISTER_OP_KERNEL_CUDNN(ReduceGPU, ReduceKernelCUDNN);

#endif

}  // namespace Shadow
