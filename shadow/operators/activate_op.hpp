#ifndef SHADOW_OPERATORS_ACTIVATE_OP_HPP
#define SHADOW_OPERATORS_ACTIVATE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ActivateOp : public Operator {
 public:
  explicit ActivateOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ActivateOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  enum {
    kPRelu = 0,
    kRelu = 1,
    kLeaky = 2,
    kSigmoid = 3,
    kSoftPlus = 4,
    kTanh = 5
  };

  int activate_type_;
  float slope_;
  bool channel_shared_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ACTIVATE_OP_HPP
