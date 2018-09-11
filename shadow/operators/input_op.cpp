#include "input_op.hpp"

namespace Shadow {

void InputOp::Forward() {}

REGISTER_OPERATOR(Input, InputOp);

}  // namespace Shadow
