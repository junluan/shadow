#include "softmax_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void SoftmaxOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
    Blas::BlasScopy(bottom->count(), bottom->data(), 0, top->mutable_data(), 0);
  }

  axis_ = bottom->canonical_index(axis_);

  outer_num_ = bottom->count(0, axis_);
  inner_num_ = bottom->count(axis_ + 1);

#if defined(USE_CUDNN)
  cudnn::setTensor4dDesc<float>(&bottom_desc_, outer_num_, bottom->shape(axis_),
                                inner_num_, 1);
  cudnn::setTensor4dDesc<float>(&top_desc_, outer_num_, bottom->shape(axis_),
                                inner_num_, 1);

  CUDNN_CHECK(cudnnSoftmaxForward(
      Kernel::cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
      cudnn::dataType<float>::one, bottom_desc_, bottom->data(),
      cudnn::dataType<float>::zero, top_desc_, top->mutable_data()));

#else
  auto scale_shape = bottom->shape();
  scale_shape[axis_] = 1;

  int temp_count = 1;
  for (auto dim : scale_shape) temp_count *= dim;
  op_ws_->GrowTempBuffer(temp_count * sizeof(float));

  scale_ = op_ws_->CreateTempBlob<float>(scale_shape, op_name_ + "_scale");

  int count = bottom->count(), channels = bottom->shape(axis_);

  Blas::ChannelMax(outer_num_, channels, inner_num_, top->data(),
                   scale_->mutable_data());
  Blas::ChannelSub(count, outer_num_, channels, inner_num_, scale_->data(),
                   top->mutable_data());
  Blas::Exp(count, top->data(), 0, top->mutable_data(), 0);
  Blas::ChannelSum(outer_num_, channels, inner_num_, top->data(),
                   scale_->mutable_data());
  Blas::ChannelDiv(count, outer_num_, channels, inner_num_, scale_->data(),
                   top->mutable_data());
#endif
}

REGISTER_OPERATOR(Softmax, SoftmaxOp);

}  // namespace Shadow
