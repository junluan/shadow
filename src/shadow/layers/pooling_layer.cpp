#include "shadow/layers/pooling_layer.hpp"
#include "shadow/util/image.hpp"

inline int pooling_out_size(int s, int size, int pad, int stride) {
  return static_cast<int>(
             std::ceil(static_cast<float>(s + 2 * pad - size) / stride)) +
         1;
}

void PoolingLayer::Reshape() {
  pool_type_ = layer_param_.pooling_param().pool();
  kernel_size_ = layer_param_.pooling_param().kernel_size();
  stride_ = layer_param_.pooling_param().stride();
  pad_ = layer_param_.pooling_param().pad();
  global_pooling_ = layer_param_.pooling_param().global_pooling();

  if (global_pooling_) {
    kernel_size_ = bottoms_[0]->shape(2);
    stride_ = 1;
    pad_ = 0;
  }

  int in_h = bottoms_[0]->shape(2), in_w = bottoms_[0]->shape(3);
  int out_h = pooling_out_size(in_h, kernel_size_, pad_, stride_);
  int out_w = pooling_out_size(in_w, kernel_size_, pad_, stride_);
  if (pad_) {
    if ((out_h - 1) * stride_ >= in_h + pad_) out_h--;
    if ((out_w - 1) * stride_ >= in_w + pad_) out_w--;
  }

  VecInt top_shape = bottoms_[0]->shape();
  top_shape[2] = out_h;
  top_shape[3] = out_w;
  tops_[0]->reshape(top_shape);

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")") << " -> "
      << kernel_size_ << "x" << kernel_size_ << "_s" << stride_ << " -> "
      << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void PoolingLayer::Forward() {
  Image::Pooling(bottoms_[0]->data(), bottoms_[0]->shape(), kernel_size_,
                 stride_, pad_, pool_type_, tops_[0]->shape(),
                 tops_[0]->mutable_data());
}

void PoolingLayer::Release() {
  bottoms_.clear();
  tops_.clear();

  // DInfo("Free PoolingLayer!");
}
