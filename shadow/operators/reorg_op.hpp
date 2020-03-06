#ifndef SHADOW_OPERATORS_REORG_OP_HPP
#define SHADOW_OPERATORS_REORG_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReorgOp : public Operator {
 public:
  ReorgOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    stride_ = get_single_argument<int>("stride", 2);
  }

  void Forward() override;

 private:
  int stride_;
};

namespace Vision {

template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data,
           Context *context);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_REORG_OP_HPP
