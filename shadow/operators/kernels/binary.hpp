#ifndef SHADOW_OPERATORS_KERNELS_BINARY_HPP_
#define SHADOW_OPERATORS_KERNELS_BINARY_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void BroadcastBinary(const T* in_data, const int* in_shape,
                     const T* scalar_data, const int* scalar_shape,
                     int operation, int num_axes, int count,
                     const int* out_shape, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

enum { kAdd = 0, kSub = 1, kMul = 2, kDiv = 3, kPow = 4, kMax = 5, kMin = 6 };

class BinaryKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int operation,
                   float scalar_value) = 0;

  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& scalar,
                   std::shared_ptr<Blob>& output, Workspace* ws, int operation,
                   bool need_broadcast) = 0;
};

template <DeviceType D>
class BinaryKernelDefault : public BinaryKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int operation, float scalar_value) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int count = input->count();

    switch (operation) {
      case kAdd:
        return Blas::Add<D, float>(count, in_data, 0, scalar_value, out_data, 0,
                                   ws->Ctx());
      case kSub:
        return Blas::Sub<D, float>(count, in_data, 0, scalar_value, out_data, 0,
                                   ws->Ctx());
      case kMul:
        return Blas::Mul<D, float>(count, in_data, 0, scalar_value, out_data, 0,
                                   ws->Ctx());
      case kDiv:
        return Blas::Div<D, float>(count, in_data, 0, scalar_value, out_data, 0,
                                   ws->Ctx());
      case kPow:
        return Blas::Pow<D, float>(count, in_data, 0, scalar_value, out_data, 0,
                                   ws->Ctx());
      case kMax:
        return Blas::Max<D, float>(count, in_data, 0, scalar_value, out_data, 0,
                                   ws->Ctx());
      case kMin:
        return Blas::Min<D, float>(count, in_data, 0, scalar_value, out_data, 0,
                                   ws->Ctx());
      default:
        LOG(FATAL) << "Unknown binary operation " << operation;
    }
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& scalar, std::shared_ptr<Blob>& output,
           Workspace* ws, int operation, bool need_broadcast) override {
    const auto* in_data = input->data<float>();
    const auto* scalar_data = scalar->data<float>();
    auto* out_data = output->mutable_data<float>();

    if (need_broadcast) {
      int num_axes = input->num_axes();

      ws->GrowTempBuffer(3 * num_axes * sizeof(int));

      auto in_shape = ws->CreateTempBlob({num_axes}, DataType::kI32);
      auto scalar_shape = ws->CreateTempBlob({num_axes}, DataType::kI32);
      auto out_shape = ws->CreateTempBlob({num_axes}, DataType::kI32);

      in_shape->set_data<int>(input->shape().data(), num_axes);
      scalar_shape->set_data<int>(scalar->shape().data(), num_axes);
      out_shape->set_data<int>(output->shape().data(), num_axes);

      Vision::BroadcastBinary<D, float>(
          in_data, in_shape->data<int>(), scalar_data,
          scalar_shape->data<int>(), operation, num_axes, output->count(),
          out_shape->data<int>(), out_data, ws->Ctx());
    } else {
      int count = input->count();

      switch (operation) {
        case kAdd:
          return Blas::Add<D, float>(count, in_data, 0, scalar_data, 0,
                                     out_data, 0, ws->Ctx());
        case kSub:
          return Blas::Sub<D, float>(count, in_data, 0, scalar_data, 0,
                                     out_data, 0, ws->Ctx());
        case kMul:
          return Blas::Mul<D, float>(count, in_data, 0, scalar_data, 0,
                                     out_data, 0, ws->Ctx());
        case kDiv:
          return Blas::Div<D, float>(count, in_data, 0, scalar_data, 0,
                                     out_data, 0, ws->Ctx());
        case kPow:
          return Blas::Pow<D, float>(count, in_data, 0, scalar_data, 0,
                                     out_data, 0, ws->Ctx());
        case kMax:
          return Blas::Max<D, float>(count, in_data, 0, scalar_data, 0,
                                     out_data, 0, ws->Ctx());
        case kMin:
          return Blas::Min<D, float>(count, in_data, 0, scalar_data, 0,
                                     out_data, 0, ws->Ctx());
        default:
          LOG(FATAL) << "Unknown binary operation " << operation;
      }
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_BINARY_HPP_
