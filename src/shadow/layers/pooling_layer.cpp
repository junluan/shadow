#include "shadow/layers/pooling_layer.hpp"
#include "shadow/util/image.hpp"

void PoolingLayer::Reshape() {
  pool_type_ = layer_param_.pooling_param().pool();
  kernel_size_ = layer_param_.pooling_param().kernel_size();
  stride_ = layer_param_.pooling_param().stride();

  int in_h = bottom_[0]->shape(2), in_w = bottom_[0]->shape(3);
  int out_h = (in_h - kernel_size_) / stride_ + 1;
  int out_w = (in_w - kernel_size_) / stride_ + 1;

  top_[0]->set_shape(bottom_[0]->shape());
  top_[0]->set_shape(2, out_h);
  top_[0]->set_shape(3, out_w);
  top_[0]->allocate_data(top_[0]->count());

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")") << " -> "
      << kernel_size_ << "x" << kernel_size_ << "_s" << stride_ << " -> "
      << Util::format_vector(top_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void PoolingLayer::Forward() {
  Image::Pooling(bottom_[0]->data(), bottom_[0]->shape(), kernel_size_, stride_,
                 pool_type_, top_[0]->shape(), top_[0]->mutable_data());
}

void PoolingLayer::Release() {
  bottom_.clear();
  top_.clear();

  // DInfo("Free PoolingLayer!");
}
