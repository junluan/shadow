#include "input_op.hpp"

namespace Shadow {

void InputOp::Forward() { DLOG(INFO) << debug_log(); }

REGISTER_OPERATOR(Input, InputOp);

}  // namespace Shadow
