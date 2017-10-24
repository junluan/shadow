#include "pooling_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

void PoolingOp::Setup() {
  pool_type_ = get_single_argument<int>("pool", 0);
  global_pooling_ = get_single_argument<bool>("global_pooling", false);
  if (!global_pooling_) {
    CHECK(has_argument("kernel_size"));
    kernel_size_ = get_single_argument<int>("kernel_size", 2);
    stride_ = get_single_argument<int>("stride", 1);
    pad_ = get_single_argument<int>("pad", 0);
  } else {
    kernel_size_ = bottoms<float>(0)->shape(2);
    stride_ = 1;
    pad_ = 0;
  }
  full_pooling_ = get_single_argument<bool>("full_pooling", true);

#if defined(USE_CUDNN)
  cudnn::createPoolingDesc<float>(&pooling_desc_, pool_type_, &mode_,
                                  kernel_size_, kernel_size_, pad_, pad_,
                                  stride_, stride_);
  cudnn::createTensor4dDesc<float>(&bottom_desc_);
  cudnn::createTensor4dDesc<float>(&top_desc_);
#endif
}

void PoolingOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int batch = bottom->shape(0), in_c = bottom->shape(1);
  int in_h = bottom->shape(2), in_w = bottom->shape(3);
  int out_h =
      pooling_out_size(in_h, kernel_size_, stride_, pad_, full_pooling_);
  int out_w =
      pooling_out_size(in_w, kernel_size_, stride_, pad_, full_pooling_);
  if (pad_) {
    if ((out_h - 1) * stride_ >= in_h + pad_) out_h--;
    if ((out_w - 1) * stride_ >= in_w + pad_) out_w--;
  }

  VecInt top_shape = bottom->shape();
  top_shape[2] = out_h;
  top_shape[3] = out_w;
  top->reshape(top_shape);

#if defined(USE_CUDNN)
  cudnn::setTensor4dDesc<float>(&bottom_desc_, batch, in_c, in_h, in_w);
  cudnn::setTensor4dDesc<float>(&top_desc_, batch, in_c, out_h, out_w);
#endif

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << kernel_size_ << "x" << kernel_size_ << "_s" << stride_ << "_p"
             << pad_ << "_t" << pool_type_ << " -> " << top->name()
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void PoolingOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

#if defined(USE_CUDNN)
  CUDNN_CHECK(cudnnPoolingForward(Kernel::cudnn_handle_, pooling_desc_,
                                  cudnn::dataType<float>::one, bottom_desc_,
                                  bottom->data(), cudnn::dataType<float>::zero,
                                  top_desc_, top->mutable_data()));

#else
  Vision::Pooling(bottom->data(), bottom->shape(), kernel_size_, stride_, pad_,
                  pool_type_, top->shape(), top->mutable_data());
#endif
}

void PoolingOp::Release() {
#if defined(USE_CUDNN)
  if (pooling_desc_ != nullptr) {
    cudnnDestroyPoolingDescriptor(pooling_desc_);
    pooling_desc_ = nullptr;
  }
  if (bottom_desc_ != nullptr) {
    cudnnDestroyTensorDescriptor(bottom_desc_);
    bottom_desc_ = nullptr;
  }
  if (top_desc_ != nullptr) {
    cudnnDestroyTensorDescriptor(top_desc_);
    top_desc_ = nullptr;
  }
#endif

  // DLOG(INFO) << "Free PoolingOp!";
}

REGISTER_OPERATOR(Pooling, PoolingOp);

}  // namespace Shadow
