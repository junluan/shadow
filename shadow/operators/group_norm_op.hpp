#ifndef SHADOW_OPERATORS_GROUP_NORM_OP_HPP
#define SHADOW_OPERATORS_GROUP_NORM_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class GroupNormOp : public Operator {
 public:
  GroupNormOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    group_ = get_single_argument<int>("group", 1);
    eps_ = get_single_argument<float>("eps", 1e-5);
  }

  void Forward() override;

 private:
  int group_;
  float eps_;
};

namespace Vision {

template <typename T>
void ComputeGroup(const T* in_data, int batch, int channel, int group,
                  T* out_data, Context* context);

template <typename T>
void SubtractMean(const T* in_data, const T* mean_data, int batch, int channel,
                  int spatial_dim, int group, T* out_data, Context* context);

template <typename T>
void DivideVariance(const T* in_data, const T* variance_data, int batch,
                    int channel, int spatial_dim, int group, float eps,
                    T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_GROUP_NORM_OP_HPP
