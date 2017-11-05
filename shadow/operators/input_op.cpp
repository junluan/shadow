#include "input_op.hpp"

namespace Shadow {

void InputOp::Reshape() {
  VecString str;
  for (int i = 0; i < tops_size(); ++i) {
    const auto *top = tops<float>(i);
    str.push_back(
        Util::format_vector(top->shape(), ",", top->name() + "(", ")"));
  }
  DLOG(INFO) << op_name_ << "(" << op_type_
             << "): " << Util::format_vector(str, ", ");
}

void InputOp::Forward() {}

REGISTER_OPERATOR(Input, InputOp);

}  // namespace Shadow
