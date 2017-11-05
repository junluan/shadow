#include "pooling_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

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

REGISTER_OPERATOR(Pooling, PoolingOp);

}  // namespace Shadow
