#ifndef SHADOW_OPERATORS_DEPTHWISE_CONV_OP_HPP
#define SHADOW_OPERATORS_DEPTHWISE_CONV_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class DepthwiseConvOp : public Operator {
 public:
  explicit DepthwiseConvOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~DepthwiseConvOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 protected:
  int num_output_, kernel_size_, stride_, pad_, dilation_, group_,
      activate_type_;
  bool bias_term_;
};

inline int conv_out_size(int dim, int kernel_size, int stride, int pad,
                         int dilation) {
  int kernel_extent = dilation * (kernel_size - 1) + 1;
  return (dim + 2 * pad - kernel_extent) / stride + 1;
}

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_DEPTHWISE_CONV_OP_HPP
