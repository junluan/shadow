#ifndef SHADOW_OPERATORS_KERNELS_CONNECTED_HPP_
#define SHADOW_OPERATORS_KERNELS_CONNECTED_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

#include "scale.hpp"

namespace Shadow {

class ConnectedKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& weight,
                   const std::shared_ptr<Blob>& bias,
                   std::shared_ptr<Blob>& output, Workspace* ws, int num_output,
                   bool bias_term, bool transpose) = 0;
};

template <DeviceType D>
class ConnectedKernelDefault : public ConnectedKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, bool bias_term,
           bool transpose) override {
    const auto* in_data = input->data<float>();
    const auto* weight_data = weight->data<float>();
    const auto* bias_data = bias_term ? bias->data<float>() : nullptr;
    auto* out_data = output->mutable_data<float>();

    int batch = input->shape(0), in_num = input->num();

    if (batch == 1) {
      if (transpose) {
        Blas::BlasSgemv<D, float>(0, num_output, in_num, 1, weight_data, 0,
                                  in_data, 0, 0, out_data, 0, ws->Ctx().get());
      } else {
        Blas::BlasSgemv<D, float>(1, in_num, num_output, 1, weight_data, 0,
                                  in_data, 0, 0, out_data, 0, ws->Ctx().get());
      }
      if (bias_term) {
        Blas::BlasSaxpy<D, float>(num_output, 1, bias_data, 0, out_data, 0,
                                  ws->Ctx().get());
      }
    } else {
      Blas::BlasSgemm<D, float>(0, transpose, batch, num_output, in_num, 1,
                                in_data, 0, weight_data, 0, 0, out_data, 0,
                                ws->Ctx().get());
      if (bias_term) {
        Vision::Bias<D, float>(out_data, output->count(), bias_data, num_output,
                               1, out_data, ws->Ctx().get());
      }
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_CONNECTED_HPP_
