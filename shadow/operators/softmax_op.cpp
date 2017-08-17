#include "softmax_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void SoftmaxOp::Setup() {
  axis_ = get_single_argument<int>("axis", 1);
  axis_ = bottoms<float>(0)->canonical_index(axis_);

#if defined(USE_CUDNN)
  cudnn::createTensor4dDesc<float>(&bottom_desc_);
  cudnn::createTensor4dDesc<float>(&top_desc_);
#endif
}

void SoftmaxOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  top->reshape(bottom->shape());

  outer_num_ = bottom->count(0, axis_);
  inner_num_ = bottom->count(axis_ + 1);

  VecInt scale_dims = bottom->shape();
  scale_dims[axis_] = 1;
  scale_.reshape(scale_dims);

#if defined(USE_CUDNN)
  cudnn::setTensor4dDesc<float>(&bottom_desc_, outer_num_, bottom->shape(axis_),
                                inner_num_, 1);
  cudnn::setTensor4dDesc<float>(&top_desc_, outer_num_, bottom->shape(axis_),
                                inner_num_, 1);
#endif

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void SoftmaxOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

#if defined(USE_CUDNN)
  CUDNN_CHECK(cudnnSoftmaxForward(
      Kernel::cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
      cudnn::dataType<float>::one, bottom_desc_, bottom->data(),
      cudnn::dataType<float>::zero, top_desc_, top->mutable_data()));

#else
  int count = bottom->count(), channels = bottom->shape(axis_);
  Blas::BlasScopy(count, bottom->data(), 0, top->mutable_data(), 0);
  Blas::ChannelMax(outer_num_, channels, inner_num_, top->data(),
                   scale_.mutable_data());
  Blas::ChannelSub(count, outer_num_, channels, inner_num_, scale_.data(),
                   top->mutable_data());
  Blas::Exp(count, top->data(), 0, top->mutable_data(), 0);
  Blas::ChannelSum(outer_num_, channels, inner_num_, top->data(),
                   scale_.mutable_data());
  Blas::ChannelDiv(count, outer_num_, channels, inner_num_, scale_.data(),
                   top->mutable_data());
#endif
}

void SoftmaxOp::Release() {
  scale_.clear();

#if defined(USE_CUDNN)
  if (bottom_desc_ != nullptr) {
    cudnnDestroyTensorDescriptor(bottom_desc_);
    bottom_desc_ = nullptr;
  }
  if (top_desc_ != nullptr) {
    cudnnDestroyTensorDescriptor(top_desc_);
    top_desc_ = nullptr;
  }
#endif

  // DLOG(INFO) << "Free SoftmaxOp!";
}

REGISTER_OPERATOR(Softmax, SoftmaxOp);

}  // namespace Shadow
