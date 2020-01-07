#ifndef SHADOW_OPERATORS_INPUT_OP_HPP
#define SHADOW_OPERATORS_INPUT_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class InputOp : public Operator {
 public:
  InputOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    for (int n = 0; n < tops_size(); ++n) {
      const auto &top_name = tops_name(n);
      const auto &top_type = tops_type(n);
      const auto &top_shape = get_repeated_argument<int>(top_name);
      if (top_type == int_id) {
        mutable_tops<int>(n)->reshape(top_shape);
      } else if (top_type == float_id) {
        mutable_tops<float>(n)->reshape(top_shape);
      } else if (top_type == uchar_id) {
        mutable_tops<unsigned char>(n)->reshape(top_shape);
      } else {
        LOG(FATAL) << "Blob " << top_name << " has unsupported type "
                   << top_type;
      }
    }
  }

  void Forward() override;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_INPUT_OP_HPP
