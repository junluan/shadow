#ifndef SHADOW_OPERATORS_SLICE_OP_HPP
#define SHADOW_OPERATORS_SLICE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class SliceOp : public Operator {
 public:
  explicit SliceOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    slice_axis_ = get_single_argument<int>("axis", 0);
    slice_point_ = get_repeated_argument<int>("slice_point");
    CHECK_GE(slice_axis_, 0);
  }

  void Forward() override;

 private:
  int slice_axis_;
  VecInt slice_point_;
};

namespace Vision {

template <typename T>
void Slice(const T *in_data, int count, int num_slices, int slice_size,
           int bottom_slice_axis, int top_slice_axis, int offset_slice_axis,
           T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SLICE_OP_HPP
