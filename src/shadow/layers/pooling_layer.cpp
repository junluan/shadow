#include "shadow/layers/pooling_layer.hpp"
#include "shadow/util/image.hpp"

void PoolingLayer::Reshape() {
  pool_type_ = layer_param_.pooling_param().pool();
  kernel_size_ = layer_param_.pooling_param().kernel_size();
  stride_ = layer_param_.pooling_param().stride();

  int in_h = bottoms_[0]->shape(2), in_w = bottoms_[0]->shape(3);
  int out_h = (in_h - kernel_size_) / stride_ + 1;
  int out_w = (in_w - kernel_size_) / stride_ + 1;

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
                 stride_, pool_type_, tops_[0]->shape(),
                 tops_[0]->mutable_data());
}

void PoolingLayer::Release() {
  bottoms_.clear();
  tops_.clear();

  // DInfo("Free PoolingLayer!");
}
