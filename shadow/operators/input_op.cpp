#include "core/operator.hpp"

namespace Shadow {

class InputOp : public Operator {
 public:
  InputOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    for (int n = 0; n < tops_size(); ++n) {
      tops(n)->reshape(get_repeated_argument<int>(tops_name(n)));
    }
  }

  void Forward() override {}
};

REGISTER_OPERATOR(Input, InputOp);

}  // namespace Shadow
