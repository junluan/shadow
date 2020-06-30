#include "core/operator.hpp"

namespace Shadow {

class InputOp : public Operator {
 public:
  InputOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {}

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::vector<std::shared_ptr<Blob>>& outputs) override {}
};

REGISTER_OPERATOR(Input, InputOp);

}  // namespace Shadow
