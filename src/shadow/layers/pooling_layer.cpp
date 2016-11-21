#include "shadow/layers/pooling_layer.hpp"
#include "shadow/util/image.hpp"

inline int pooling_out_size(int dim, int kernel_size, int stride, int pad) {
  return static_cast<int>(std::ceil(
             static_cast<float>(dim + 2 * pad - kernel_size) / stride)) +
         1;
}

void PoolingLayer::Setup(VecBlob *blobs) {
  Layer::Setup(blobs);

  const auto &pooling_param = layer_param_.pooling_param();

  pool_type_ = pooling_param.pool();
  global_pooling_ = pooling_param.global_pooling();
  if (!global_pooling_) {
    CHECK(pooling_param.has_kernel_size());
    kernel_size_ = pooling_param.kernel_size();
    stride_ = pooling_param.stride();
    pad_ = pooling_param.pad();
  } else {
    kernel_size_ = bottoms_[0]->shape(2);
    stride_ = 1;
    pad_ = 0;
  }
}

void PoolingLayer::Reshape() {
  int in_h = bottoms_[0]->shape(2), in_w = bottoms_[0]->shape(3);
  int out_h = pooling_out_size(in_h, kernel_size_, stride_, pad_);
  int out_w = pooling_out_size(in_w, kernel_size_, stride_, pad_);
  if (pad_) {
    if ((out_h - 1) * stride_ >= in_h + pad_) out_h--;
    if ((out_w - 1) * stride_ >= in_w + pad_) out_w--;
  }

  VecInt top_shape = bottoms_[0]->shape();
  top_shape[2] = out_h;
  top_shape[3] = out_w;
  tops_[0]->reshape(top_shape);

  DLOG(INFO) << layer_name_ << ": "
             << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")")
             << " -> " << kernel_size_ << "x" << kernel_size_ << "_s" << stride_
             << " -> " << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
}

void PoolingLayer::Forward() {
  Image::Pooling(bottoms_[0]->data(), bottoms_[0]->shape(), kernel_size_,
                 stride_, pad_, pool_type_, tops_[0]->shape(),
                 tops_[0]->mutable_data());
}

void PoolingLayer::Release() {
  bottoms_.clear();
  tops_.clear();

  // DLOG(INFO) << "Free PoolingLayer!";
}
