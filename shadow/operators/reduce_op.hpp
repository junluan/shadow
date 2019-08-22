#ifndef SHADOW_OPERATORS_REDUCE_OP_HPP
#define SHADOW_OPERATORS_REDUCE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReduceOp : public Operator {
 public:
  ReduceOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", 0);
    axes_ = get_repeated_argument<int>("axes");
    keep_dims_ = get_single_argument<bool>("keep_dims", true);

#if defined(USE_CUDNN)
    cudnn::createReduceDesc<float>(&reduce_desc_);
    cudnn::createTensorDesc<float>(&bottom_desc_);
    cudnn::createTensorDesc<float>(&top_desc_);
#endif
  }
  ~ReduceOp() override {
#if defined(USE_CUDNN)
    if (reduce_desc_ != nullptr) {
      cudnnDestroyReduceTensorDescriptor(reduce_desc_);
      reduce_desc_ = nullptr;
    }
    if (bottom_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(bottom_desc_);
      bottom_desc_ = nullptr;
    }
    if (top_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(top_desc_);
      top_desc_ = nullptr;
    }
#endif
  }

  void Forward() override;

  enum { kProd = 0, kSum = 1, kMax = 2, kMin = 3, kAvg = 4 };

 private:
  int operation_;
  bool keep_dims_;
  VecInt axes_, bottom_shape_, top_shape_, list_value_, offset_value_;

#if defined(USE_CUDNN)
  cudnnReduceTensorDescriptor_t reduce_desc_ = nullptr;
  cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
#endif
};

namespace Vision {

template <typename T>
void Reduce(const T *in_data, const int *list_data, const int *offset_data,
            int num_list, int operation, int count, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_REDUCE_OP_HPP
