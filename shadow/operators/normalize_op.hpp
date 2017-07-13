#ifndef SHADOW_OPERATORS_NORMALIZE_OP_HPP
#define SHADOW_OPERATORS_NORMALIZE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class NormalizeOp : public Operator {
 public:
  explicit NormalizeOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~NormalizeOp() { Release(); }

  virtual void Setup() override;
  virtual void Reshape() override;
  virtual void Forward() override;
  virtual void Release() override;

 private:
  bool across_spatial_, channel_shared_;
  int spatial_dim_;
  float scale_;

  BlobF norm_, buffer_;
  BlobF sum_channel_multiplier_, sum_spatial_multiplier_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_NORMALIZE_OP_HPP
