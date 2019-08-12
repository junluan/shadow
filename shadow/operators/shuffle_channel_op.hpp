#ifndef SHADOW_OPERATORS_SHUFFLE_CHANNEL_OP_HPP
#define SHADOW_OPERATORS_SHUFFLE_CHANNEL_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ShuffleChannelOp : public Operator {
 public:
  ShuffleChannelOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    group_ = get_single_argument<int>("group", 0);
    CHECK_GT(group_, 0) << "group must be larger than 0";
  }

  void Forward() override;

 private:
  int group_;
};

namespace Vision {

template <typename T>
void ShuffleChannel(const T *in_data, int batch, int channel, int spatial_dim,
                    int group, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SHUFFLE_CHANNEL_OP_HPP
