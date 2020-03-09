#include "shuffle_channel_op.hpp"

namespace Shadow {

void ShuffleChannelOp::Forward() {
  const auto bottom = bottoms(0);
  auto top = tops(0);

  CHECK_NE(bottom, top);

  int batch = bottom->shape(0), channel = bottom->shape(1),
      spatial_dim = bottom->count(2);

  CHECK_EQ(channel % group_, 0);

  top->reshape(bottom->shape());

  Vision::ShuffleChannel(bottom->data<float>(), batch, channel, spatial_dim,
                         group_, top->mutable_data<float>(), ws_->Ctx());
}

REGISTER_OPERATOR(ShuffleChannel, ShuffleChannelOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void ShuffleChannel(const T *in_data, int batch, int channel, int spatial_dim,
                    int group, T *out_data, Context *context) {
  int num = channel * spatial_dim;
  int group_column = channel / group;
  for (int b = 0; b < batch; ++b, in_data += num, out_data += num) {
    for (int c = 0; c < channel; ++c) {
      int c_out = (c % group_column) * group + c / group_column;
      memcpy(out_data + c_out * spatial_dim, in_data + c * spatial_dim,
             spatial_dim * sizeof(T));
    }
  }
}

template void ShuffleChannel(const float *, int, int, int, int, float *,
                             Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
