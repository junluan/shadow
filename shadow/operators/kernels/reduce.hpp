#ifndef SHADOW_OPERATORS_KERNELS_REDUCE_HPP_
#define SHADOW_OPERATORS_KERNELS_REDUCE_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Reduce(const T* in_data, const int* list_data, const int* offset_data,
            int num_list, int operation, int count, T* out_data,
            Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

enum {
  kProd = 0,
  kSum = 1,
  kMax = 2,
  kMin = 3,
  kAvg = 4,
  kLpNorm1 = 5,
  kLpNorm2 = 6
};

class ReduceKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int operation,
                   const VecInt& axes) = 0;
};

template <DeviceType D>
class ReduceKernelDefault : public ReduceKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int operation, const VecInt& axes) override {
    int num_axes = input->num_axes();

    if (in_shape_ != input->shape()) {
      in_shape_ = input->shape();

      VecInt shape_acc(1, 1);
      for (int n = num_axes - 1; n > 0; --n) {
        shape_acc.insert(shape_acc.begin(), input->shape(n) * shape_acc[0]);
      }
      list_value_ = {0};
      for (int n = static_cast<int>(axes.size()) - 1; n >= 0; --n) {
        int axis = axes[n], num_list = static_cast<int>(list_value_.size());
        for (int k = 1; k < input->shape(axis); ++k) {
          for (int j = 0; j < num_list; ++j) {
            list_value_.push_back(list_value_[j] + k * shape_acc[axis]);
          }
        }
      }
      offset_value_.clear();
      for (int i = 0; i < output->count(); ++i) {
        int offset = 0, cc = i;
        for (int n = num_axes - 1; n >= 0; --n) {
          int dim = output->shape(n);
          offset += (cc % dim) * shape_acc[n];
          cc /= dim;
        }
        offset_value_.push_back(offset);
      }
    }

    int num_list = static_cast<int>(list_value_.size());
    int num_offset = static_cast<int>(offset_value_.size());

    ws->GrowTempBuffer((num_list + num_offset) * sizeof(int));

    auto list = ws->CreateTempBlob({num_list}, DataType::kI32);
    auto offset = ws->CreateTempBlob({num_offset}, DataType::kI32);

    list->set_data<int>(list_value_.data(), num_list);
    offset->set_data<int>(offset_value_.data(), num_offset);

    Vision::Reduce<D, float>(
        input->data<float>(), list->data<int>(), offset->data<int>(), num_list,
        operation, output->count(), output->mutable_data<float>(), ws->Ctx());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }

 private:
  VecInt in_shape_, list_value_, offset_value_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_REDUCE_HPP_
