#ifndef SHADOW_OPERATORS_KERNELS_ELTWISE_HPP_
#define SHADOW_OPERATORS_KERNELS_ELTWISE_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

namespace Shadow {

class EltwiseKernel : public Kernel {
 public:
  virtual void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
                   std::shared_ptr<Blob>& output, Workspace* ws, int operation,
                   const VecFloat& coeff) = 0;
};

template <DeviceType D>
class EltwiseKernelDefault : public EltwiseKernel {
 public:
  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::shared_ptr<Blob>& output, Workspace* ws, int operation,
           const VecFloat& coeff) override {
    auto* out_data = output->mutable_data<float>();

    int count = inputs[0]->count();

    switch (operation) {
      case kProd:
        Blas::Mul<D, float>(count, inputs[0]->data<float>(), 0,
                            inputs[1]->data<float>(), 0, out_data, 0,
                            ws->Ctx());
        for (int n = 2; n < inputs.size(); ++n) {
          Blas::Mul<D, float>(count, out_data, 0, inputs[n]->data<float>(), 0,
                              out_data, 0, ws->Ctx());
        }
        break;
      case kSum:
        Blas::Set<D, float>(count, 0, out_data, 0, ws->Ctx());
        for (int n = 0; n < inputs.size(); ++n) {
          Blas::BlasSaxpy<D, float>(count, coeff[n], inputs[n]->data<float>(),
                                    0, out_data, 0, ws->Ctx());
        }
        break;
      case kMax:
        Blas::Max<D, float>(count, inputs[0]->data<float>(), 0,
                            inputs[1]->data<float>(), 0, out_data, 0,
                            ws->Ctx());
        for (int n = 2; n < inputs.size(); ++n) {
          Blas::Max<D, float>(count, out_data, 0, inputs[n]->data<float>(), 0,
                              out_data, 0, ws->Ctx());
        }
        break;
      case kMin:
        Blas::Min<D, float>(count, inputs[0]->data<float>(), 0,
                            inputs[1]->data<float>(), 0, out_data, 0,
                            ws->Ctx());
        for (int n = 2; n < inputs.size(); ++n) {
          Blas::Min<D, float>(count, out_data, 0, inputs[n]->data<float>(), 0,
                              out_data, 0, ws->Ctx());
        }
        break;
      default:
        LOG(FATAL) << "Unknown eltwise operation " << operation;
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }

 private:
  enum { kProd = 0, kSum = 1, kMax = 2, kMin = 3 };
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_ELTWISE_HPP_
