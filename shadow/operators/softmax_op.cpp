#include "softmax_op.hpp"
#include "core/blas.hpp"

namespace Shadow {

void SoftmaxOp::Setup(VecBlobF *blobs) {
  Operator::Setup(blobs);

  const auto &softmax_param = op_param_.softmax_param();

  axis_ = bottoms_[0]->canonical_index(softmax_param.axis());

#if defined(USE_CUDNN)
  cudnn::createTensor4dDesc<float>(&bottom_desc_);
  cudnn::createTensor4dDesc<float>(&top_desc_);
#endif
}

void SoftmaxOp::Reshape() {
  tops_[0]->reshape(bottoms_[0]->shape());

  outer_num_ = bottoms_[0]->count(0, axis_);
  inner_num_ = bottoms_[0]->count(axis_ + 1);

  VecInt scale_dims = bottoms_[0]->shape();
  scale_dims[axis_] = 1;
  scale_.reshape(scale_dims);

#if defined(USE_CUDNN)
  cudnn::setTensor4dDesc<float>(&bottom_desc_, outer_num_,
                                bottoms_[0]->shape(axis_), inner_num_, 1);
  cudnn::setTensor4dDesc<float>(&top_desc_, outer_num_,
                                bottoms_[0]->shape(axis_), inner_num_, 1);
#endif

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void SoftmaxOp::Forward() {
#if defined(USE_CUDNN)
  CUDNN_CHECK(cudnnSoftmaxForward(
      Kernel::cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
      cudnn::dataType<float>::one, bottom_desc_, bottoms_[0]->data(),
      cudnn::dataType<float>::zero, top_desc_, tops_[0]->mutable_data()));

#else
  int count = bottoms_[0]->count(), channels = bottoms_[0]->shape(axis_);
  Blas::BlasScopy(count, bottoms_[0]->data(), 0, tops_[0]->mutable_data(), 0);
  Blas::ChannelMax(outer_num_, channels, inner_num_, tops_[0]->data(),
                   scale_.mutable_data());
  Blas::ChannelSub(count, outer_num_, channels, inner_num_, scale_.data(),
                   tops_[0]->mutable_data());
  Blas::Exp(count, tops_[0]->data(), 0, tops_[0]->mutable_data(), 0);
  Blas::ChannelSum(outer_num_, channels, inner_num_, tops_[0]->data(),
                   scale_.mutable_data());
  Blas::ChannelDiv(count, outer_num_, channels, inner_num_, scale_.data(),
                   tops_[0]->mutable_data());
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

REGISTER_OP_CLASS(Softmax);

}  // namespace Shadow
