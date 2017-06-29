#ifndef SHADOW_OPERATORS_BATCH_NORM_OP_HPP
#define SHADOW_OPERATORS_BATCH_NORM_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BatchNormOp : public Operator {
 public:
  explicit BatchNormOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~BatchNormOp() { Release(); }

  void Setup();
  void Reshape();
  void Forward();
  void Release();

 private:
  bool use_global_stats_;
  float scale_;
  int channels_, spatial_dim_;

  BlobF mean_, variance_, temp_;
  BlobF sum_batch_multiplier_, sum_spatial_multiplier_, batch_by_channel_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BATCH_NORM_OP_HPP
