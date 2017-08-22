#ifndef SHADOW_OPERATORS_NORMALIZE_OP_HPP
#define SHADOW_OPERATORS_NORMALIZE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class NormalizeOp : public Operator {
 public:
  explicit NormalizeOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~NormalizeOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  bool across_spatial_, channel_shared_;
  int spatial_dim_;
  float scale_;

  BlobF *norm_ = nullptr, *buffer_ = nullptr;
  BlobF *sum_channel_multiplier_ = nullptr, *sum_spatial_multiplier_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_NORMALIZE_OP_HPP
