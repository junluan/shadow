#include "softmax_op.hpp"

namespace Shadow {

void SoftmaxOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0,
                    op_ws_->Ctx()->blas_handle());
  }

  axis_ = bottom->canonical_index(axis_);

  int outer_num = bottom->count(0, axis_), inner_num = bottom->count(axis_ + 1);

#if defined(USE_CUDNN)
  cudnn::setTensor4dDesc<float>(&bottom_top_desc_, outer_num,
                                bottom->shape(axis_), inner_num, 1);

  CUDNN_CHECK(cudnnSoftmaxForward(
      cudnnHandle_t(op_ws_->Ctx()->cudnn_handle()), CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL, cudnn::dataType<float>::one, bottom_top_desc_,
      bottom->data(), cudnn::dataType<float>::zero, bottom_top_desc_,
      top->mutable_data()));

#else
  auto scale_shape = bottom->shape();
  scale_shape[axis_] = 1;

  int temp_count = 1;
  for (auto dim : scale_shape) temp_count *= dim;
  op_ws_->GrowTempBuffer(temp_count, sizeof(float));

  auto *scale = op_ws_->CreateTempBlob<float>(scale_shape, op_name_ + "/scale");

  int count = bottom->count(), channels = bottom->shape(axis_);

  Blas::ChannelMax(outer_num, channels, inner_num, top->data(),
                   scale->mutable_data());
  Blas::ChannelSub(count, outer_num, channels, inner_num, scale->data(),
                   top->mutable_data());
  Blas::Exp(count, top->data(), 0, top->mutable_data(), 0);
  Blas::ChannelSum(outer_num, channels, inner_num, top->data(),
                   scale->mutable_data());
  Blas::ChannelDiv(count, outer_num, channels, inner_num, scale->data(),
                   top->mutable_data());
#endif
}

REGISTER_OPERATOR(Softmax, SoftmaxOp);

}  // namespace Shadow
