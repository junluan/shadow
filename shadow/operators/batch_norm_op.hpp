#ifndef SHADOW_OPERATORS_BATCH_NORM_OP_HPP
#define SHADOW_OPERATORS_BATCH_NORM_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BatchNormOp : public Operator {
 public:
  explicit BatchNormOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~BatchNormOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  bool use_global_stats_;
  float eps_, scale_;
  int channels_, spatial_dim_;

  BlobF *mean_ = nullptr, *variance_ = nullptr, *temp_ = nullptr;
  BlobF *sum_batch_multiplier_ = nullptr, *sum_spatial_multiplier_ = nullptr,
        *batch_by_channel_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BATCH_NORM_OP_HPP
