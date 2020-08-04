#ifndef SHADOW_OPERATORS_KERNELS_UNARY_HPP_
#define SHADOW_OPERATORS_KERNELS_UNARY_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

namespace Shadow {

enum {
  kAbs = 0,
  kSquare = 1,
  kSqrt = 2,
  kLog = 3,
  kExp = 4,
  kSin = 5,
  kCos = 6,
  kTan = 7,
  kAsin = 8,
  kAcos = 9,
  kAtan = 10,
  kFloor = 11,
  kCeil = 12,
  kNeg = 13,
  kReciprocal = 14
};

class UnaryKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   int operation) = 0;
};

template <DeviceType D>
class UnaryKernelDefault : public UnaryKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int operation) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int count = input->count();

    switch (operation) {
      case kAbs:
        return Blas::Abs<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kSquare:
        return Blas::Square<D, float>(count, in_data, 0, out_data, 0,
                                      ws->Ctx());
      case kSqrt:
        return Blas::Sqrt<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kLog:
        return Blas::Log<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kExp:
        return Blas::Exp<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kSin:
        return Blas::Sin<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kCos:
        return Blas::Cos<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kTan:
        return Blas::Tan<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kAsin:
        return Blas::Asin<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kAcos:
        return Blas::Acos<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kAtan:
        return Blas::Atan<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kFloor:
        return Blas::Floor<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kCeil:
        return Blas::Ceil<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kNeg:
        return Blas::Neg<D, float>(count, in_data, 0, out_data, 0, ws->Ctx());
      case kReciprocal:
        return Blas::Reciprocal<D, float>(count, in_data, 0, out_data, 0,
                                          ws->Ctx());
      default:
        LOG(FATAL) << "Unknown unary operation " << operation;
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_UNARY_HPP_
